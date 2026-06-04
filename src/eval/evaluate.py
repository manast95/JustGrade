"""Full evaluation pipeline: metrics, confusion matrix, model-card report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eval.metrics import bootstrap_qwk, compute_all
from src.utils.config import get
from src.utils.logging import get_logger

logger = get_logger(__name__)

REPORTS = Path("reports")
FIGURES = REPORTS / "figures"


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    label_col: str = "human_score",
    model_name: str = "model",
    y_pred: np.ndarray | None = None,
) -> dict:
    """Compute all metrics and persist to reports/metrics.json.

    Args:
        model: fitted BaselineScorer or TransformerScorer with .predict(df).
        test_df: test split DataFrame.
        label_col: ground-truth label column.
        model_name: key under which to store results ('baseline' or 'transformer').
        y_pred: optional precomputed predictions — pass to avoid a second,
            redundant predict() call (the transformer's predict is expensive).

    Returns:
        dict of metric_name → value.
    """
    lo, hi = get("label_range", [1, 5])
    if y_pred is None:
        y_pred = model.predict(test_df)
    y_true = test_df[label_col].to_numpy()
    metrics = compute_all(y_true, y_pred, min_rating=lo, max_rating=hi)

    # Bootstrap CI on the headline metric
    boot = bootstrap_qwk(
        y_true, y_pred,
        n_boot=get("eval.bootstrap_n", 1000),
        ci=get("eval.bootstrap_ci", 0.95),
        seed=get("seed", 42),
        min_rating=lo, max_rating=hi,
    )
    metrics.update(boot)

    REPORTS.mkdir(exist_ok=True)
    metrics_path = REPORTS / "metrics.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing[model_name] = {"label_col": label_col, **metrics}
    metrics_path.write_text(json.dumps(existing, indent=2))

    logger.info(
        f"{model_name} — QWK={metrics['qwk']:.4f} "
        f"[{boot['qwk_ci_low']:.4f}, {boot['qwk_ci_high']:.4f}]  "
        f"MAE={metrics['mae']:.4f}"
    )
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    save_path: str | None = None,
) -> Path:
    """Save a normalised confusion-matrix heatmap.

    Returns the path the figure was saved to.
    """
    lo, hi = get("label_range", [1, 5])
    labels = list(range(lo, hi + 1))
    n = len(labels)

    # Build confusion matrix manually (avoids sklearn import for a simple op)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t) - lo][int(p) - lo] += 1

    # Row-normalise, guarding against true classes with zero samples (→ avoid NaN)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        vmin=0, vmax=1, ax=ax,
    )
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("True score")
    ax.set_title(f"Confusion matrix — {model_name} (row-normalised)")
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = Path(save_path or FIGURES / f"confusion_matrix_{model_name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to '{out}'")
    return out


def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    save_path: str | None = None,
) -> Path:
    """Reliability curve: mean true score for each predicted score bucket.

    A well-calibrated scorer sits on the diagonal — when it predicts a 4, the
    true score averages ~4. Deviations reveal systematic over/under-scoring at
    specific score levels.

    Returns the path the figure was saved to.
    """
    lo, hi = get("label_range", [1, 5])
    yt = np.asarray(y_true, dtype=float)
    preds = np.round(np.asarray(y_pred)).astype(int).clip(lo, hi)

    xs, means, counts = [], [], []
    for s in range(lo, hi + 1):
        mask = preds == s
        if mask.any():
            xs.append(s)
            means.append(float(yt[mask].mean()))
            counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([lo, hi], [lo, hi], "--", color="grey", label="Perfect calibration")
    ax.plot(xs, means, "o-", color="#1976d2", label="Observed")
    # Annotate each point with its support count
    for x, m, c in zip(xs, means, counts):
        ax.annotate(f"n={c}", (x, m), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color="#555")

    ax.set_xlabel("Predicted score")
    ax.set_ylabel("Mean true score")
    ax.set_title(f"Calibration — {model_name}")
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(lo - 0.3, hi + 0.3)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = Path(save_path or FIGURES / f"calibration_{model_name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Calibration curve saved to '{out}'")
    return out


def write_model_card(metrics_path: str = "reports/metrics.json") -> str:
    """Generate a markdown model-card table comparing all logged models.

    Returns the markdown string (also printed to stdout).
    """
    data = json.loads(Path(metrics_path).read_text())

    rows = []
    metric_keys = ["qwk", "mae", "pearson", "spearman", "within_one_acc"]
    metric_labels = {
        "qwk":           "QWK (↑)",
        "mae":           "MAE (↓)",
        "pearson":       "Pearson r (↑)",
        "spearman":      "Spearman ρ (↑)",
        "within_one_acc": "Within-1 acc (↑)",
    }

    model_names = [k for k in data if k not in ("label_col",)]
    header = "| Metric | " + " | ".join(m.title() for m in model_names) + " |"
    sep    = "|--------|" + "|".join("-------:" for _ in model_names) + "|"
    rows.append(header)
    rows.append(sep)

    for key in metric_keys:
        vals = [data[m].get(key, float("nan")) for m in model_names]
        # Bold the best value for each metric
        if key == "mae":
            best_idx = int(np.argmin(vals))
        else:
            best_idx = int(np.argmax(vals))
        cells = []
        for i, v in enumerate(vals):
            formatted = f"{v:.4f}"
            cells.append(f"**{formatted}**" if i == best_idx else formatted)
        rows.append(f"| {metric_labels[key]} | " + " | ".join(cells) + " |")

    # QWK 95% CI row (if bootstrap data is present)
    if any("qwk_ci_low" in data[m] for m in model_names):
        ci_cells = []
        for m in model_names:
            md = data[m]
            if "qwk_ci_low" in md and "qwk_ci_high" in md:
                ci_cells.append(f"[{md['qwk_ci_low']:.4f}, {md['qwk_ci_high']:.4f}]")
            else:
                ci_cells.append("—")
        ci_level = next((data[m].get("ci_level") for m in model_names
                         if data[m].get("ci_level")), 0.95)
        rows.append(f"| QWK {int(ci_level * 100)}% CI | " + " | ".join(ci_cells) + " |")

    card = "\n".join(rows)
    return card
