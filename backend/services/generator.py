"""
LLM Generator for Knowledge-Shift RAG.

Generates answers using an OpenAI-compatible API given
retrieved context and a question.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI, AzureOpenAI

from backend.config import config


GENERATION_SYSTEM_PROMPT = """You are an educational question-answering assistant.

Your task is to answer multiple-choice questions based ONLY on the provided evidence.

Rules:
1. Base your answer strictly on the provided evidence.
2. Do not invent facts not present in the evidence.
3. If the evidence is insufficient, say so explicitly.
4. Return your answer as a JSON object with these fields:
   - "answer": the letter of the correct choice (A, B, C, or D)
   - "answer_text": the full text of the chosen answer
   - "reasoning": brief explanation of why this answer is correct based on the evidence
   - "source_ids": list of source chunk IDs that support your answer
   - "confidence": "high", "medium", or "low"
"""

GENERATION_USER_TEMPLATE = """Evidence:
{context}

Question: {question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer using ONLY the evidence above. Respond with a JSON object."""


class Generator:
    """LLM-based answer generator."""

    def __init__(self, llm_config: Any = None):
        cfg = llm_config or config.llm
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens

        if getattr(cfg, "is_azure", False):
            self.client = AzureOpenAI(
                api_key=cfg.api_key,
                api_version=cfg.azure_api_version,
                azure_endpoint=cfg.azure_endpoint,
                max_retries=5,
            )
            self.model = cfg.azure_deployment or self.model
        else:
            self.client = OpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
            )

    def generate(
        self,
        question: str,
        choices: dict[str, str],
        context: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate an answer for a multiple-choice question.

        Args:
            question: The question text.
            choices: Dict with keys 'A', 'B', 'C', 'D' mapping to choice text.
            context: Retrieved context string.
            system_prompt: Optional override for the system prompt.

        Returns:
            Parsed JSON response or a fallback dict with raw text.
        """
        user_prompt = GENERATION_USER_TEMPLATE.format(
            context=context,
            question=question,
            choice_a=choices.get("A", ""),
            choice_b=choices.get("B", ""),
            choice_c=choices.get("C", ""),
            choice_d=choices.get("D", ""),
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt or GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_text = response.choices[0].message.content.strip()

            # Try to parse JSON from the response
            return self._parse_response(raw_text)

        except Exception as e:
            return {
                "answer": "E",  # error marker
                "answer_text": "",
                "reasoning": f"LLM call failed: {str(e)}",
                "source_ids": [],
                "confidence": "none",
                "raw": str(e),
                "error": True,
            }

    def _parse_response(self, raw_text: str) -> dict[str, Any]:
        """Try to parse JSON from the LLM response."""
        # Try direct JSON parse
        try:
            result = json.loads(raw_text)
            result["raw"] = raw_text
            result["error"] = False
            return result
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        if "```json" in raw_text:
            try:
                json_str = raw_text.split("```json")[1].split("```")[0].strip()
                result = json.loads(json_str)
                result["raw"] = raw_text
                result["error"] = False
                return result
            except (json.JSONDecodeError, IndexError):
                pass

        if "```" in raw_text:
            try:
                json_str = raw_text.split("```")[1].split("```")[0].strip()
                result = json.loads(json_str)
                result["raw"] = raw_text
                result["error"] = False
                return result
            except (json.JSONDecodeError, IndexError):
                pass

        # Fallback: extract letter answer using regex
        import re
        match = re.search(r'"answer"\s*:\s*"([A-D])"', raw_text)
        letter = match.group(1) if match else "E"

        return {
            "answer": letter,
            "answer_text": "",
            "reasoning": "Could not parse structured response",
            "source_ids": [],
            "confidence": "low",
            "raw": raw_text,
            "error": False,
            "parse_fallback": True,
        }
