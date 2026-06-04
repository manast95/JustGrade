"""Pure evaluation metrics for ordinal response scoring."""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


def _to_int(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.round(arr).astype(int).clip(lo, hi)


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_rating: int = 1,
    max_rating: int = 5,
) -> float:
    """Quadratic Weighted Kappa — field-standard ordinal agreement metric.

    Penalises large disagreements quadratically. Perfect scorer → 1.0,
    random scoring → ~0.0, systematic inverse scoring → negative.
    """
    labels = list(range(min_rating, max_rating + 1))
    y_true_int = _to_int(np.asarray(y_true), min_rating, max_rating)
    y_pred_int = _to_int(np.asarray(y_pred), min_rating, max_rating)
    return float(cohen_kappa_score(y_true_int, y_pred_int,
                                   weights="quadratic", labels=labels))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = pearsonr(np.asarray(y_true, dtype=float),
                    np.asarray(y_pred, dtype=float))
    return float(r)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho, _ = spearmanr(np.asarray(y_true, dtype=float),
                       np.asarray(y_pred, dtype=float))
    return float(rho)


def within_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions within 1 point of the true label."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


def bootstrap_qwk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    min_rating: int = 1,
    max_rating: int = 5,
) -> dict[str, float]:
    """Bootstrap confidence interval for QWK via case resampling.

    Resamples (y_true, y_pred) pairs with replacement n_boot times and reports
    the mean and the central ``ci`` percentile interval of the QWK distribution.
    This turns a point estimate (0.93) into a defensible range (0.93 [0.91, 0.95]).

    Returns:
        dict with qwk_mean, qwk_ci_low, qwk_ci_high, ci_level, n_boot.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    rng = np.random.default_rng(seed)

    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = quadratic_weighted_kappa(yt[idx], yp[idx], min_rating, max_rating)

    lo_pct = (1.0 - ci) / 2.0 * 100.0
    hi_pct = (1.0 + ci) / 2.0 * 100.0
    return {
        "qwk_mean":    float(stats.mean()),
        "qwk_ci_low":  float(np.percentile(stats, lo_pct)),
        "qwk_ci_high": float(np.percentile(stats, hi_pct)),
        "ci_level":    ci,
        "n_boot":      n_boot,
    }


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_rating: int = 1,
    max_rating: int = 5,
) -> dict[str, float]:
    """Compute the full metric suite and return as a dict."""
    return {
        "qwk":              quadratic_weighted_kappa(y_true, y_pred, min_rating, max_rating),
        "mae":              mean_absolute_error(y_true, y_pred),
        "pearson":          pearson_correlation(y_true, y_pred),
        "spearman":         spearman_correlation(y_true, y_pred),
        "within_one_acc":   within_one_accuracy(y_true, y_pred),
    }
