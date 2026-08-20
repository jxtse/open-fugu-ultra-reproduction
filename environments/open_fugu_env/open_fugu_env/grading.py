"""Deterministic answer grading: math-verify first, normalized exact match fallback."""

from __future__ import annotations

import re


def _normalize(text: str) -> str:
    text = text.strip().strip("$")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    return text


def grade_answer(prediction: str, ground_truth: str) -> bool:
    """Return True when prediction matches ground truth.

    Tries math_verify's symbolic/numeric equivalence first (handles LaTeX,
    fractions, sympy-equivalent forms); falls back to whitespace-insensitive
    exact match for non-mathematical strings.
    """
    prediction = (prediction or "").strip()
    ground_truth = (ground_truth or "").strip()
    if not prediction:
        return False

    if _normalize(prediction) == _normalize(ground_truth):
        return True

    try:
        from math_verify import parse, verify

        def _candidates(text: str):
            candidates = [parse(text)]
            if "$" not in text:
                candidates.append(parse(f"${text}$"))
            return [item for item in candidates if item]

        gold_candidates = _candidates(ground_truth)
        pred_candidates = _candidates(prediction)
        if any(
            verify(gold, pred)
            for gold in gold_candidates
            for pred in pred_candidates
        ):
            return True
    except Exception:
        pass

    return False


__all__ = ["grade_answer"]
