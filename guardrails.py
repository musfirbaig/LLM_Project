"""Guardrails for bank QA assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""


INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"disregard\s+(all|previous)",
    r"reveal\s+(system|hidden)\s+prompt",
    r"act\s+as\s+.*(hacker|attacker|jailbreak)",
    r"you\s+are\s+now",
]

DISALLOWED_PATTERNS = [
    r"how\s+to\s+(hack|phish|scam|steal)",
    r"bypass\s+(security|verification|otp)",
    r"make\s+(a\s+)?bomb",
    r"fraud",
    r"money\s+laundering",
]

MAX_QUERY_CHARS = 800


def inspect_query(query: str) -> GuardrailResult:
    text = (query or "").strip()
    if not text:
        return GuardrailResult(False, "Please provide a valid question.")

    if len(text) > MAX_QUERY_CHARS:
        return GuardrailResult(False, "Your question is too long. Please shorten it.")

    lowered = text.lower()

    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            return GuardrailResult(False, "I can only answer banking product questions safely.")

    for pat in DISALLOWED_PATTERNS:
        if re.search(pat, lowered):
            return GuardrailResult(False, "I cannot help with harmful or disallowed requests.")

    return GuardrailResult(True)


def retrieval_confidence_from_distance(distance: float) -> float:
    # Milvus score from similarity_search_with_score is typically a distance (lower is better).
    return 1.0 / (1.0 + max(distance, 0.0))


def post_filter(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return "I could not generate a reliable answer right now. Please try again."

    # Prevent accidental policy leakage phrases.
    banned_fragments = ["system prompt", "hidden instruction", "developer message"]
    lowered = text.lower()
    if any(fragment in lowered for fragment in banned_fragments):
        return "I can help with NUST Bank product information, but I cannot provide internal instructions."

    return text
