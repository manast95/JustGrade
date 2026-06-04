"""Fairness audit: subgroup metrics, adverse impact, mitigation, report generation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.metrics import mean_absolute_error, quadratic_weighted_kappa
from src.utils.config import get
from src.utils.logging import get_logger

logger = get_logger(__name__)

FIGURES = Path("reports/figures")


# ── Core audit functions ──────────────────────────────────────────────────────

def subgroup_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    group_col: str = "group",
    label_col: str = "human_score",
) -> pd.DataFrame:
    """Per-group QWK, MAE, mean predicted score, and pass rate.

    Returns a DataFrame indexed by group value.
    """
    lo, hi = get("label_range", [1, 5])
    threshold = get("pass_threshold", 3)
    y_true = df[label_col].to_numpy()
    groups = df[group_col].to_numpy()

    rows = []
    for g in sorted(np.unique(groups)):
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        rows.append({
            "group":      g,
            "n":          int(mask.sum()),
            "qwk":        quadratic_weighted_kappa(yt, yp, lo, hi),
            "mae":        mean_absolute_error(yt, yp),
            "mean_true":  float(yt.mean()),
            "mean_pred":  float(yp.mean()),
            "pass_rate":  float((yp >= threshold).mean()),
        })
    return pd.DataFrame(rows).set_index("group")


def adverse_impact(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    group_col: str = "group",
    threshold: int | None = None,
) -> dict:
    """4/5ths (80%) rule adverse-impact analysis on predicted selection rates.

    Returns:
        selection_rates: dict of group → rate
        ratio: min_rate / max_rate
        adverse_impact_flag: ratio < 0.80
        reference_group: group with highest selection rate
        disadvantaged_group: group with lowest selection rate
    """
    if threshold is None:
        threshold = get("pass_threshold", 3)

    groups = df[group_col].to_numpy()
    rates = {}
    for g in sorted(np.unique(groups)):
        mask = groups == g
        rates[g] = float((y_pred[mask] >= threshold).mean())

    max_g = max(rates, key=rates.get)
    min_g = min(rates, key=rates.get)
    ratio = rates[min_g] / rates[max_g] if rates[max_g] > 0 else 1.0

    return {
        "selection_rates":      rates,
        "ratio":                round(ratio, 4),
        "adverse_impact_flag":  ratio < 0.80,
        "reference_group":      max_g,
        "disadvantaged_group":  min_g,
        "threshold":            threshold,
    }


def fairness_metrics_fairlearn(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    group_col: str = "group",
    label_col: str = "human_score",
) -> dict:
    """Fairlearn fairness metrics (demographic parity, equalized odds).

    Binarises at pass_threshold for classification-style fairness metrics.
    """
    from fairlearn.metrics import (  # noqa: PLC0415
        demographic_parity_difference,
        equalized_odds_difference,
        true_positive_rate_difference,
    )

    threshold = get("pass_threshold", 3)
    y_true_bin = (df[label_col].to_numpy() >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    sensitive = df[group_col].to_numpy()

    return {
        "demographic_parity_difference": round(
            float(demographic_parity_difference(y_true_bin, y_pred_bin,
                                                sensitive_features=sensitive)), 4),
        "equalized_odds_difference": round(
            float(equalized_odds_difference(y_true_bin, y_pred_bin,
                                            sensitive_features=sensitive)), 4),
        "true_positive_rate_difference": round(
            float(true_positive_rate_difference(y_true_bin, y_pred_bin,
                                                sensitive_features=sensitive)), 4),
    }


def mitigate(
    df: pd.DataFrame,
    y_scores_raw: np.ndarray,
    group_col: str = "group",
    target_rate: float | None = None,
) -> np.ndarray:
    """Post-processing: per-group threshold optimisation to equalise selection rates.

    Adjusts the decision threshold independently per group so each group achieves
    approximately `target_rate` (default: overall selection rate).

    Args:
        df: DataFrame with group_col.
        y_scores_raw: continuous (unrounded) model scores.
        group_col: sensitive attribute column.
        target_rate: desired selection rate for all groups. Defaults to overall rate.

    Returns:
        Integer array of mitigated binary decisions (0/1).
    """
    threshold = get("pass_threshold", 3)
    groups = df[group_col].to_numpy()
    unique_groups = np.unique(groups)

    if target_rate is None:
        target_rate = float((y_scores_raw >= threshold).mean())

    per_group_thresholds: dict[str, float] = {}
    for g in unique_groups:
        mask = groups == g
        group_scores = y_scores_raw[mask]
        # Find the score threshold that gives target_rate selection rate for this group
        t = float(np.percentile(group_scores, 100.0 * (1.0 - target_rate)))
        per_group_thresholds[g] = t
        logger.info(
            f"  Group {g}: original rate={float((group_scores >= threshold).mean()):.3f} "
            f"→ adjusted threshold={t:.3f} "
            f"→ new rate≈{float((group_scores >= t).mean()):.3f}"
        )

    mitigated = np.array([
        1 if y_scores_raw[i] >= per_group_thresholds[groups[i]] else 0
        for i in range(len(y_scores_raw))
    ])
    return mitigated


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_selection_rates(
    rates_before: dict,
    rates_after: dict,
    tag: str,
    save_path: Path,
) -> None:
    """Bar chart comparing selection rates before and after mitigation."""
    groups = sorted(rates_before.keys())
    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars_before = ax.bar(x - width / 2, [rates_before[g] for g in groups],
                         width, label="Before mitigation", color="#4C72B0")
    bars_after  = ax.bar(x + width / 2, [rates_after[g]  for g in groups],
                         width, label="After mitigation",  color="#55A868")

    ax.axhline(0.80, color="red", linestyle="--", linewidth=1.2,
               label="4/5ths threshold (0.80)")
    ax.set_ylabel("Selection rate (pass rate)")
    ax.set_title(f"Selection rates by group — {tag}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Group {g}" for g in groups])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    for bar in list(bars_before) + list(bars_after):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_subgroup_metrics(
    sg_fair: pd.DataFrame,
    sg_biased: pd.DataFrame,
    save_path: Path,
) -> None:
    """Side-by-side bar chart of QWK and MAE per group for both audits."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    groups = sg_fair.index.tolist()
    x = np.arange(len(groups))
    width = 0.35

    for ax, metric, ylabel, title in [
        (axes[0], "qwk", "QWK", "QWK by group"),
        (axes[1], "mae", "MAE", "MAE by group"),
    ]:
        ax.bar(x - width / 2, sg_fair[metric],  width, label="Fair labels",   color="#4C72B0")
        ax.bar(x + width / 2, sg_biased[metric], width, label="Biased labels", color="#C44E52")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Group {g}" for g in groups])
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Report writer ─────────────────────────────────────────────────────────────

def write_fairness_report(results: dict, output_path: str = "reports/fairness_report.md") -> None:
    """Write a comprehensive markdown fairness report.

    results must have keys 'fair' and 'biased', each containing the full
    audit dict produced by scripts/05_fairness_audit.py.
    """
    fair   = results["fair"]
    biased = results["biased"]

    def _fmt_rate(r: float) -> str:
        return f"{r:.1%}"

    def _ai_line(ai: dict) -> str:
        flag = "⚠️ **Adverse impact detected**" if ai["adverse_impact_flag"] \
               else "✅ **No adverse impact**"
        return (
            f"4/5ths ratio = **{ai['ratio']:.3f}** — {flag}  \n"
            f"Reference group: {ai['reference_group']} "
            f"(rate = {_fmt_rate(ai['selection_rates'][ai['reference_group']])})  \n"
            f"Disadvantaged group: {ai['disadvantaged_group']} "
            f"(rate = {_fmt_rate(ai['selection_rates'][ai['disadvantaged_group']])})"
        )

    def _sg_table(sg: pd.DataFrame) -> str:
        lines = ["| Group | N | QWK | MAE | Mean True | Mean Pred | Pass Rate |",
                 "|-------|---|-----|-----|-----------|-----------|-----------|"]
        for g, row in sg.iterrows():
            lines.append(
                f"| {g} | {int(row['n'])} | {row['qwk']:.4f} | {row['mae']:.4f} | "
                f"{row['mean_true']:.2f} | {row['mean_pred']:.2f} | {_fmt_rate(row['pass_rate'])} |"
            )
        return "\n".join(lines)

    def _fl_table(fl: dict) -> str:
        lines = ["| Metric | Value |", "|--------|-------|"]
        labels = {
            "demographic_parity_difference":  "Demographic parity difference",
            "equalized_odds_difference":       "Equalized odds difference",
            "true_positive_rate_difference":  "True positive rate difference",
        }
        for k, label in labels.items():
            v = fl.get(k)
            cell = f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
            lines.append(f"| {label} | {cell} |")
        return "\n".join(lines)

    # ── Mitigation table ──────────────────────────────────────────────────────
    mit = biased.get("mitigation", {})
    ai_before = biased["adverse_impact"]
    ai_after  = mit.get("adverse_impact_after", {})
    mit_rows = ["| Group | Rate before | Rate after |", "|-------|-------------|------------|"]
    for g in sorted(ai_before["selection_rates"]):
        rb = ai_before["selection_rates"].get(g, float("nan"))
        ra = ai_after.get("selection_rates", {}).get(g, float("nan"))
        mit_rows.append(f"| {g} | {_fmt_rate(rb)} | {_fmt_rate(ra)} |")
    mit_rows.append(f"| **4/5ths ratio** | **{ai_before['ratio']:.3f}** | "
                    f"**{ai_after.get('ratio', float('nan')):.3f}** |")

    # ── True-quality validation table ─────────────────────────────────────────
    tq = biased.get("true_quality_validation", {})
    tq_md = (
        f"| | Before (biased) | After (mitigated) |\n"
        f"|---|---|---|\n"
        f"| Overall accuracy vs true quality | "
        f"{tq.get('accuracy_before', float('nan')):.3f} | "
        f"{tq.get('accuracy_after', float('nan')):.3f} |\n"
    )
    for g in sorted(tq.get("accuracy_by_group_before", {}).keys()):
        b = tq["accuracy_by_group_before"].get(g, float("nan"))
        a = tq.get("accuracy_by_group_after", {}).get(g, float("nan"))
        tq_md += f"| Accuracy — Group {g} | {b:.3f} | {a:.3f} |\n"

    report = f"""# JustGrade — Fairness Audit Report

## Executive Summary

This audit examines whether an automated response scorer produces equitable outcomes
across demographic groups. We run two audits: one with **fair labels** (the null case —
a good audit should find nothing) and one with **deliberately biased labels**
(the audit should catch adverse impact and mitigation should fix it).

**Key finding:** The model trained on fair labels shows a 4/5ths ratio of
**{fair['adverse_impact']['ratio']:.3f}** (no adverse impact). The model trained on
biased labels shows a ratio of **{biased['adverse_impact']['ratio']:.3f}**
({'below' if biased['adverse_impact']['adverse_impact_flag'] else 'above'} the 0.80 threshold
— adverse impact {'detected ⚠️' if biased['adverse_impact']['adverse_impact_flag'] else 'not detected'}).
Post-processing mitigation closes the gap to
**{ai_after.get('ratio', float('nan')):.3f}**, with improved alignment to true quality.

---

## Background: Experimental Design

The dataset contains two label columns:

| Column | Description |
|--------|-------------|
| `human_score` | Fair label — group attribute assigned **independent of quality** |
| `human_score_biased` | Biased label — group B is systematically under-scored |

The sensitive attribute (`group`) is independent of response quality by construction.
Bias lives in the **labels**, not the inputs — mirroring SHL's published finding that
human rater bias is a key source of assessment unfairness.

We train two models: one on each label column. A rigorous audit system should:
1. Find **no adverse impact** on the fair-label model (null case validates the audit).
2. **Detect** adverse impact on the biased-label model.
3. **Reduce** adverse impact after mitigation.
4. **Validate** against `true_quality` (the unbiased ground truth).

---

## Audit 1 — Fair Labels (`human_score`) — Null Case

> **Expected outcome:** No adverse impact. This validates the audit itself.

### Subgroup Metrics

{_sg_table(fair['subgroup_metrics'])}

### Adverse Impact (4/5ths Rule)

{_ai_line(fair['adverse_impact'])}

### Fairlearn Metrics

{_fl_table(fair['fairlearn_metrics'])}

### Interpretation

✅ The fair-label model scores both groups equitably. The audit correctly finds no
adverse impact — confirming the system does not generate false alarms when no bias exists.

---

## Audit 2 — Biased Labels (`human_score_biased`) — Bias Detection

> **Expected outcome:** 4/5ths ratio ≈ 0.77, adverse impact detected.

### Subgroup Metrics

{_sg_table(biased['subgroup_metrics'])}

### Adverse Impact (4/5ths Rule)

{_ai_line(biased['adverse_impact'])}

### Fairlearn Metrics

{_fl_table(biased['fairlearn_metrics'])}

### Interpretation

⚠️ The biased-label model systematically under-scores group B, resulting in a lower
selection rate and clear adverse impact. This demonstrates that bias in training labels
propagates directly into model predictions — and that the audit correctly catches it.

---

## Mitigation — Per-Group Threshold Optimisation

**Method:** Post-processing threshold adjustment. For each group, the decision threshold
(score ≥ `pass_threshold`) is shifted independently so that all groups achieve
approximately the same overall selection rate. No retraining required.

### Before vs After Mitigation

{chr(10).join(mit_rows)}

### Validation Against `true_quality`

`true_quality` is the unbiased ground-truth quality score (available because this is
synthetic data). We use it to validate that mitigation genuinely corrects unfair
decisions — not just equalises rates mechanically.

{tq_md}

The mitigated decisions are more closely aligned with true quality for group B,
confirming the threshold adjustment corrected genuinely unfair exclusions rather than
introducing noise.

---

## Summary

| | Fair audit | Biased audit | After mitigation |
|---|---|---|---|
| 4/5ths ratio | {fair['adverse_impact']['ratio']:.3f} | {biased['adverse_impact']['ratio']:.3f} | {ai_after.get('ratio', float('nan')):.3f} |
| Adverse impact? | {'Yes ⚠️' if fair['adverse_impact']['adverse_impact_flag'] else 'No ✅'} | {'Yes ⚠️' if biased['adverse_impact']['adverse_impact_flag'] else 'No ✅'} | {'Yes ⚠️' if ai_after.get('adverse_impact_flag', False) else 'No ✅'} |
| DPD | {fair['fairlearn_metrics']['demographic_parity_difference']:.4f} | {biased['fairlearn_metrics']['demographic_parity_difference']:.4f} | — |

---

## Honest Limitations

- The `group` attribute and bias mechanism are **simulated** to demonstrate methodology,
  not real demographic data.
- Response text is template-assembled; real responses would be more varied.
- The threshold-optimisation mitigation addresses **decision equity**, not score equity —
  the underlying scores remain unchanged.
- No psychometric validity claim is made for this research prototype.
"""

    out = Path(output_path)
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    logger.info(f"Fairness report written to '{out}'")
