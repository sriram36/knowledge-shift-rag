"""
Knowledge-Repair module for Knowledge-Shift RAG (System C).

When the critic classifies an answer as UNCERTAIN or UNSUPPORTED,
this module:
1. Reformulates the query for additional evidence retrieval.
2. Retrieves additional evidence.
3. Combines original + additional evidence.
4. Asks an LLM verifier to produce a corrected answer.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI, AzureOpenAI

from backend.config import config
from backend.services.retriever import Retriever


REFORMULATE_SYSTEM_PROMPT = """You are a query reformulation assistant.

Given a question and critique feedback about why an answer was unsupported,
generate a reformulated search query that would help find the correct evidence.

Respond with a JSON object:
{
  "reformulated_query": "...",
  "search_focus": "brief description of what to look for"
}
"""

REPAIR_SYSTEM_PROMPT = """You are a knowledge-repair assistant for educational question answering.

You have two sets of evidence:
1. Original evidence (from the first retrieval)
2. Additional evidence (from a follow-up retrieval after the initial answer was flagged as uncertain/unsupported)

Your task:
1. Review both sets of evidence carefully.
2. Determine if the additional evidence resolves the uncertainty.
3. Produce a corrected answer to the question.

Respond with a JSON object:
{
  "answer": "A, B, C, or D",
  "answer_text": "the text of the chosen answer",
  "reasoning": "explain how the additional evidence helped (or didn't)",
  "repair_successful": true/false,
  "source_ids": ["list of supporting chunk IDs"]
}
"""

REPAIR_USER_TEMPLATE = """Original Evidence:
{original_context}

Additional Evidence:
{additional_context}

Question: {question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Previous Answer: {previous_answer} ({previous_answer_text})
Critique Status: {critique_status}
Critique Reason: {critique_reason}

Using ALL available evidence, provide the corrected answer. Respond with a JSON object."""


class KnowledgeRepair:
    """Lightweight knowledge-repair flow."""

    def __init__(self, retriever: Retriever, llm_config: Any = None):
        self.retriever = retriever
        cfg = llm_config or config.llm
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens

        if getattr(cfg, "is_azure", False):
            self.client = AzureOpenAI(
                api_key=cfg.api_key,
                api_version=cfg.azure_api_version,
                azure_endpoint=cfg.azure_endpoint,
            )
            self.model = cfg.azure_deployment or self.model
        else:
            self.client = OpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
            )

    def reformulate_query(
        self,
        question: str,
        critique_reason: str,
    ) -> str:
        """Generate a reformulated search query based on critique feedback."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": REFORMULATE_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Original question: {question}\n"
                        f"Critique feedback: {critique_reason}\n\n"
                        "Generate a reformulated search query."
                    )},
                ],
            )

            raw = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(raw)
                return parsed.get("reformulated_query", question)
            except json.JSONDecodeError:
                # If we can't parse, use the raw text as query
                return raw[:500] if len(raw) > 500 else raw

        except Exception:
            # Fallback: just use the original question with added context
            return f"{question} {critique_reason}"

    def repair(
        self,
        question: str,
        choices: dict[str, str],
        original_context: str,
        previous_answer: str,
        previous_answer_text: str,
        critique_status: str,
        critique_reason: str,
        additional_top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Execute the knowledge-repair flow.

        1. Reformulate query
        2. Retrieve additional evidence
        3. Generate corrected answer

        Returns a dict with the repair result.
        """
        # Step 1: Reformulate
        reformulated_query = self.reformulate_query(question, critique_reason)

        # Step 2: Additional retrieval
        additional_results = self.retriever.retrieve(reformulated_query, top_k=additional_top_k)
        additional_context = "\n\n".join(
            f"[Additional Source {i} | {r.chunk_id} | {r.subject}]\n{r.text}"
            for i, r in enumerate(additional_results, 1)
        )

        # Step 3: Generate corrected answer
        user_prompt = REPAIR_USER_TEMPLATE.format(
            original_context=original_context,
            additional_context=additional_context,
            question=question,
            choice_a=choices.get("A", ""),
            choice_b=choices.get("B", ""),
            choice_c=choices.get("C", ""),
            choice_d=choices.get("D", ""),
            previous_answer=previous_answer,
            previous_answer_text=previous_answer_text,
            critique_status=critique_status,
            critique_reason=critique_reason,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_text = response.choices[0].message.content.strip()
            result = self._parse_response(raw_text)

        except Exception as e:
            result = {
                "answer": previous_answer,
                "answer_text": previous_answer_text,
                "reasoning": f"Repair LLM call failed: {str(e)}",
                "repair_successful": False,
                "source_ids": [],
                "raw": str(e),
                "error": True,
            }

        # Attach repair metadata
        result["reformulated_query"] = reformulated_query
        result["additional_chunks"] = [r.to_dict() for r in additional_results]
        result["repair_triggered"] = True

        return result

    def _parse_response(self, raw_text: str) -> dict[str, Any]:
        """Parse the repair response."""
        # Try direct JSON
        for attempt_text in [raw_text]:
            try:
                result = json.loads(attempt_text)
                result["raw"] = raw_text
                result["error"] = False
                return result
            except json.JSONDecodeError:
                pass

        # Try markdown extraction
        if "```json" in raw_text:
            try:
                json_str = raw_text.split("```json")[1].split("```")[0].strip()
                result = json.loads(json_str)
                result["raw"] = raw_text
                result["error"] = False
                return result
            except (json.JSONDecodeError, IndexError):
                pass

        # Fallback
        import re
        match = re.search(r'"answer"\s*:\s*"([A-D])"', raw_text)
        letter = match.group(1) if match else "E"

        return {
            "answer": letter,
            "answer_text": "",
            "reasoning": "Parsed from raw repair response (fallback)",
            "repair_successful": False,
            "source_ids": [],
            "raw": raw_text,
            "error": False,
            "parse_fallback": True,
        }
