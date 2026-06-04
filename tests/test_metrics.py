"""Unit tests for src/eval/metrics.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    compute_all,
    mean_absolute_error,
    pearson_correlation,
    quadratic_weighted_kappa,
    spearman_correlation,
    within_one_accuracy,
)

# ── QWK ──────────────────────────────────────────────────────────────────────

def test_qwk_perfect_agreement():
    y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)


def test_qwk_worst_case_negative():
    y_true = np.array([1, 1, 1, 1, 1, 5, 5, 5, 5, 5])
    y_pred = np.array([5, 5, 5, 5, 5, 1, 1, 1, 1, 1])
    assert quadratic_weighted_kappa(y_true, y_pred) < 0


def test_qwk_clips_out_of_range():
    """Predictions outside [1,5] must be clipped, not raise."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([0, 1, 3, 5, 6])   # 0 and 6 are out of range
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    assert -1.0 <= qwk <= 1.0


def test_qwk_near_perfect():
    """Off-by-one on one sample should still give high QWK."""
    y_true = np.array([1, 2, 3, 4, 5] * 10)
    y_pred = np.array([1, 2, 3, 4, 5] * 9 + [1, 2, 3, 4, 4])  # last score off by 1
    assert quadratic_weighted_kappa(y_true, y_pred) > 0.95


# ── MAE ──────────────────────────────────────────────────────────────────────

def test_mae_zero():
    y = np.array([1, 2, 3])
    assert mean_absolute_error(y, y) == pytest.approx(0.0)


def test_mae_known_value():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([2, 3, 4, 5, 5])   # each off by 1 except last
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.8)


# ── Pearson / Spearman ───────────────────────────────────────────────────────

def test_pearson_perfect_positive():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pearson_correlation(y, y) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert pearson_correlation(y_true, y_pred) == pytest.approx(-1.0)


def test_spearman_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman_correlation(y, y) == pytest.approx(1.0)


# ── Within-one accuracy ───────────────────────────────────────────────────────

def test_within_one_all_exact():
    y = np.array([1, 2, 3, 4, 5])
    assert within_one_accuracy(y, y) == pytest.approx(1.0)


def test_within_one_mixed():
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 3])   # last off by 2 → 80% within-1
    assert within_one_accuracy(y_true, y_pred) == pytest.approx(0.8)


def test_within_one_none():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([4, 5, 4])   # all off by > 1
    assert within_one_accuracy(y_true, y_pred) == pytest.approx(0.0)


# ── compute_all ───────────────────────────────────────────────────────────────

def test_compute_all_keys():
    y = np.array([1, 2, 3, 4, 5] * 4)
    result = compute_all(y, y)
    assert set(result.keys()) == {"qwk", "mae", "pearson", "spearman", "within_one_acc"}


def test_compute_all_perfect():
    y = np.array([1, 2, 3, 4, 5] * 4)
    result = compute_all(y, y)
    assert result["qwk"]            == pytest.approx(1.0)
    assert result["mae"]            == pytest.approx(0.0)
    assert result["pearson"]        == pytest.approx(1.0)
    assert result["spearman"]       == pytest.approx(1.0)
    assert result["within_one_acc"] == pytest.approx(1.0)
