"""
Self-Critique module for Knowledge-Shift RAG.

Takes an initial LLM answer, the question, and retrieved context,
then classifies the answer as SUPPORTED / UNCERTAIN / UNSUPPORTED.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from backend.config import config


CRITIC_SYSTEM_PROMPT = """You are a critical evaluator for a question-answering system.

Your task is to evaluate whether a given answer is properly SUPPORTED by the provided evidence.

Classify the answer into one of three categories:
- SUPPORTED: The answer is clearly and directly supported by the evidence.
- UNCERTAIN: The evidence partially supports the answer but there are gaps or ambiguities.
- UNSUPPORTED: The answer contradicts the evidence, or the evidence does not contain the information needed.

Respond with a JSON object containing:
- "status": one of "SUPPORTED", "UNCERTAIN", "UNSUPPORTED"
- "reason": a brief explanation of your classification
- "conflicting_info": any information in the evidence that contradicts the answer (empty string if none)
"""

CRITIC_USER_TEMPLATE = """Evidence:
{context}

Question: {question}

Given Answer: {answer} ({answer_text})

Evaluate whether this answer is supported by the evidence. Respond with a JSON object."""


class Critic:
    """Self-critique module that evaluates answer quality."""

    def __init__(self, llm_config: Any = None):
        cfg = llm_config or config.llm
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens

        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )

    def critique(
        self,
        question: str,
        answer: str,
        answer_text: str,
        context: str,
    ) -> dict[str, Any]:
        """
        Evaluate an answer against retrieved evidence.

        Args:
            question: The original question.
            answer: The letter answer (A/B/C/D).
            answer_text: The text of the chosen answer.
            context: The retrieved evidence context.

        Returns:
            Dict with status, reason, and conflicting_info.
        """
        user_prompt = CRITIC_USER_TEMPLATE.format(
            context=context,
            question=question,
            answer=answer,
            answer_text=answer_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_text = response.choices[0].message.content.strip()
            return self._parse_response(raw_text)

        except Exception as e:
            return {
                "status": "UNCERTAIN",
                "reason": f"Critique call failed: {str(e)}",
                "conflicting_info": "",
                "raw": str(e),
                "error": True,
            }

    def _parse_response(self, raw_text: str) -> dict[str, Any]:
        """Parse the critic response."""
        # Try direct JSON
        try:
            result = json.loads(raw_text)
            result.setdefault("status", "UNCERTAIN")
            result.setdefault("reason", "")
            result.setdefault("conflicting_info", "")
            result["raw"] = raw_text
            result["error"] = False
            # Normalize status
            result["status"] = result["status"].upper()
            if result["status"] not in ("SUPPORTED", "UNCERTAIN", "UNSUPPORTED"):
                result["status"] = "UNCERTAIN"
            return result
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown
        if "```json" in raw_text:
            try:
                json_str = raw_text.split("```json")[1].split("```")[0].strip()
                result = json.loads(json_str)
                result.setdefault("status", "UNCERTAIN")
                result.setdefault("reason", "")
                result.setdefault("conflicting_info", "")
                result["raw"] = raw_text
                result["error"] = False
                result["status"] = result["status"].upper()
                if result["status"] not in ("SUPPORTED", "UNCERTAIN", "UNSUPPORTED"):
                    result["status"] = "UNCERTAIN"
                return result
            except (json.JSONDecodeError, IndexError):
                pass

        # Fallback: look for status keywords
        upper = raw_text.upper()
        if "UNSUPPORTED" in upper:
            status = "UNSUPPORTED"
        elif "SUPPORTED" in upper:
            status = "SUPPORTED"
        else:
            status = "UNCERTAIN"

        return {
            "status": status,
            "reason": "Parsed from raw text (fallback)",
            "conflicting_info": "",
            "raw": raw_text,
            "error": False,
            "parse_fallback": True,
        }
