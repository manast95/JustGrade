"""Prediction confidence + human-review routing.

Mirrors SHL's *Augmented Intelligence* / human-in-the-loop pattern: the model
scores confidently where it can, and defers ambiguous cases to a human rater
rather than forcing a potentially-wrong automated decision.

Confidence is derived from how close the continuous (pre-rounding) score sits to
its nearest integer. A raw score of 4.02 → very confident "4"; a raw score of
3.5 → maximally ambiguous between 3 and 4, so confidence ≈ 0.
"""
from __future__ import annotations

import numpy as np

from src.utils.config import get


def prediction_confidence(raw_scores: np.ndarray) -> np.ndarray:
    """Map continuous scores to a confidence in [0, 1].

    confidence = 1 - 2·|raw − round(raw)|

    The distance to the nearest integer is in [0, 0.5], so confidence spans
    [0, 1]: 0 at the .5 midpoint (maximally ambiguous), 1 exactly on an integer.
    """
    raw = np.asarray(raw_scores, dtype=float)
    distance = np.abs(raw - np.round(raw))      # 0 … 0.5
    return np.clip(1.0 - 2.0 * distance, 0.0, 1.0)


def needs_review(
    raw_scores: np.ndarray,
    min_confidence: float | None = None,
) -> np.ndarray:
    """Boolean mask of predictions whose confidence is below the threshold.

    Args:
        raw_scores: continuous (unrounded) model scores.
        min_confidence: threshold; defaults to config.review.min_confidence.

    Returns:
        Boolean array — True where the case should be routed to human review.
    """
    if min_confidence is None:
        min_confidence = get("review.min_confidence", 0.5)
    return prediction_confidence(raw_scores) < min_confidence


def review_rate(raw_scores: np.ndarray, min_confidence: float | None = None) -> float:
    """Fraction of predictions that would be routed to human review."""
    return float(needs_review(raw_scores, min_confidence).mean())


def confidence_band(confidence: float) -> str:
    """Human-readable label for a single confidence value."""
    if confidence >= 0.7:
        return "High"
    if confidence >= 0.5:
        return "Medium"
    return "Low — route to human review"
