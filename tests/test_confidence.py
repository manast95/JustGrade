"""Unit tests for src/eval/confidence.py and metrics.bootstrap_qwk."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.confidence import (
    confidence_band,
    needs_review,
    prediction_confidence,
    review_rate,
)
from src.eval.metrics import bootstrap_qwk


# ── Confidence ────────────────────────────────────────────────────────────────

def test_confidence_max_on_integer():
    assert prediction_confidence([4.0])[0] == pytest.approx(1.0)


def test_confidence_min_at_midpoint():
    assert prediction_confidence([3.5])[0] == pytest.approx(0.0)


def test_confidence_linear():
    # 0.25 from nearest int → 1 - 2*0.25 = 0.5
    assert prediction_confidence([3.25])[0] == pytest.approx(0.5)


def test_confidence_array():
    out = prediction_confidence([1.0, 2.5, 3.1])
    assert out.shape == (3,)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)


def test_needs_review_flags_low_confidence():
    raw = np.array([4.0, 3.5, 2.05])   # conf: 1.0, 0.0, 0.9
    mask = needs_review(raw, min_confidence=0.5)
    assert list(mask) == [False, True, False]


def test_review_rate():
    raw = np.array([4.0, 3.5, 2.5, 1.0])  # two at midpoint → review
    assert review_rate(raw, min_confidence=0.5) == pytest.approx(0.5)


def test_confidence_band_labels():
    assert confidence_band(0.9).startswith("High")
    assert confidence_band(0.6).startswith("Medium")
    assert confidence_band(0.2).startswith("Low")


# ── Bootstrap QWK ─────────────────────────────────────────────────────────────

def test_bootstrap_qwk_perfect():
    y = np.array([1, 2, 3, 4, 5] * 20)
    res = bootstrap_qwk(y, y, n_boot=200, seed=0)
    assert res["qwk_mean"] == pytest.approx(1.0)
    assert res["qwk_ci_low"] == pytest.approx(1.0)
    assert res["qwk_ci_high"] == pytest.approx(1.0)


def test_bootstrap_qwk_interval_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(1, 6, 300)
    # Predictions correlated with truth but noisy
    y_pred = np.clip(y_true + rng.integers(-1, 2, 300), 1, 5)
    res = bootstrap_qwk(y_true, y_pred, n_boot=300, seed=1)
    assert res["qwk_ci_low"] <= res["qwk_mean"] <= res["qwk_ci_high"]
    assert 0.0 <= res["qwk_ci_low"] <= res["qwk_ci_high"] <= 1.0


def test_bootstrap_qwk_reproducible():
    y_true = np.array([1, 2, 3, 4, 5] * 10)
    y_pred = np.array([1, 2, 3, 4, 4] * 10)
    a = bootstrap_qwk(y_true, y_pred, n_boot=100, seed=42)
    b = bootstrap_qwk(y_true, y_pred, n_boot=100, seed=42)
    assert a == b
