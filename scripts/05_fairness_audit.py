"""Phase 5: Fairness audit on BOTH label columns — detect, mitigate, report.

The key insight: bias lives in the LABELS (human_score_biased), not the text.
Since group is assigned independently of text quality, an ML model cannot learn
the group-specific bias from text alone. We therefore audit LABEL DISTRIBUTIONS
(the scoring system's output) as well as model predictions, and mitigate via
post-processing on the label-based decisions.

This mirrors real-world scenarios where human rater bias taints the ground truth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import load_split, get_label
from src.eval.fairness import (
    _plot_selection_rates,
    _plot_subgroup_metrics,
    adverse_impact,
    fairness_metrics_fairlearn,
    mitigate,
    subgroup_metrics,
    write_fairness_report,
)
from src.models.baseline import BaselineScorer
from src.utils.config import get
from src.utils.logging import get_logger

logger = get_logger("fairness_audit")
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)


def _accuracy_vs_true(y_binary: np.ndarray, df, group_col: str) -> dict:
    """Accuracy of binary pass/fail decisions vs true_quality (unbiased ground truth)."""
    threshold = get("pass_threshold", 3)
    y_true_bin = (df["true_quality"].to_numpy() >= threshold).astype(int)
    groups = df[group_col].to_numpy()
    return {
        "accuracy_overall": float((y_binary == y_true_bin).mean()),
        "by_group": {
            g: float((y_binary[groups == g] == y_true_bin[groups == g]).mean())
            for g in sorted(np.unique(groups))
        },
    }


def run_audit(
    model: BaselineScorer,
    test_df,
    label_col: str,
    tag: str,
) -> dict:
    """Run a complete fairness audit for one label column.

    Audits BOTH the label distribution (the scoring system's raw outputs) AND
    model predictions, then mitigates if adverse impact is detected in labels.
    """
    group_col = get("fairness.group_col", "group")

    logger.info(f"\n{'='*60}\nAudit: label={label_col!r}  [{tag}]\n{'='*60}")

    # Label scores (what the scoring SYSTEM actually awarded) and model predictions
    y_labels = get_label(test_df, label_col).to_numpy().astype(float)
    y_pred   = model.predict(test_df).astype(float)
    y_scores = model.predict_raw(test_df)

    # ── Subgroup metrics (labels as ground truth) ─────────────────────────────
    sg = subgroup_metrics(test_df, y_pred, group_col=group_col, label_col=label_col)
    logger.info(f"\nSubgroup metrics (model vs {label_col}):\n{sg.to_string()}")

    # ── Adverse impact on LABEL distribution ─────────────────────────────────
    ai_labels = adverse_impact(test_df, y_labels, group_col=group_col)
    flag_l = "⚠️  ADVERSE IMPACT" if ai_labels["adverse_impact_flag"] else "✅  No adverse impact"
    logger.info(
        f"\nAdverse impact — LABEL distribution ({label_col}):"
        f"\n  ratio={ai_labels['ratio']:.3f}  {flag_l}"
        f"\n  Selection rates: {ai_labels['selection_rates']}"
    )

    # ── Adverse impact on MODEL PREDICTIONS ──────────────────────────────────
    ai_pred = adverse_impact(test_df, y_pred, group_col=group_col)
    flag_p = "⚠️  ADVERSE IMPACT" if ai_pred["adverse_impact_flag"] else "✅  No adverse impact"
    logger.info(
        f"\nAdverse impact — MODEL predictions:"
        f"\n  ratio={ai_pred['ratio']:.3f}  {flag_p}"
        f"\n  Selection rates: {ai_pred['selection_rates']}"
    )

    # ── Fairlearn metrics (label-based) ───────────────────────────────────────
    fl = fairness_metrics_fairlearn(test_df, y_pred, group_col=group_col, label_col=label_col)
    logger.info(f"\nFairlearn (model vs {label_col}):  {fl}")

    result = {
        "label_col":            label_col,
        "tag":                  tag,
        "subgroup_metrics":     sg,
        "adverse_impact":       ai_labels,     # primary: label-based
        "adverse_impact_pred":  ai_pred,        # secondary: prediction-based
        "fairlearn_metrics":    fl,
    }

    # ── Mitigation (only when label bias is detected) ─────────────────────────
    if ai_labels["adverse_impact_flag"]:
        logger.info(
            "\nApplying per-group threshold optimisation on continuous model scores …\n"
            "  (continuous scores allow thresholds to land between integers, "
            "hitting the exact target rate)"
        )
        # Target: match the reference group's (unbiased) selection rate
        target_rate = ai_labels["selection_rates"][ai_labels["reference_group"]]

        # Mitigate on continuous model scores (not discrete label integers)
        y_mitigated = mitigate(test_df, y_scores, group_col=group_col,
                               target_rate=target_rate)
        # y_mitigated is binary (0/1), threshold=1 counts 1s as "pass"
        ai_after = adverse_impact(test_df, y_mitigated, group_col=group_col,
                                  threshold=1)

        flag_after = "⚠️  Still flagged" if ai_after["adverse_impact_flag"] else "✅  Resolved"
        logger.info(
            f"After mitigation:  ratio={ai_after['ratio']:.3f}  {flag_after}"
            f"\n  Selection rates: {ai_after['selection_rates']}"
        )

        # Validate against true_quality
        y_before_bin = (y_labels >= get("pass_threshold", 3)).astype(int)
        acc_before = _accuracy_vs_true(y_before_bin,  test_df, group_col)
        acc_after  = _accuracy_vs_true(y_mitigated,   test_df, group_col)

        logger.info(
            f"\nValidation vs true_quality:"
            f"\n  Accuracy before: overall={acc_before['accuracy_overall']:.3f}  "
            f"per-group={acc_before['by_group']}"
            f"\n  Accuracy after:  overall={acc_after['accuracy_overall']:.3f}  "
            f"per-group={acc_after['by_group']}"
        )

        result["mitigation"] = {"adverse_impact_after": ai_after}
        result["true_quality_validation"] = {
            "accuracy_before":          acc_before["accuracy_overall"],
            "accuracy_after":           acc_after["accuracy_overall"],
            "accuracy_by_group_before": acc_before["by_group"],
            "accuracy_by_group_after":  acc_after["by_group"],
        }

        _plot_selection_rates(
            rates_before=ai_labels["selection_rates"],
            rates_after=ai_after["selection_rates"],
            tag=tag,
            save_path=FIGURES / f"fairness_selection_rates_{tag.lower()}.png",
        )

    return result


def main() -> None:
    group_col = get("fairness.group_col", "group")

    logger.info("Loading data …")
    train_df = load_split("train")
    test_df  = load_split("test")

    # ── Audit 1: FAIR labels ──────────────────────────────────────────────────
    logger.info("\nLoading pre-trained baseline (human_score) …")
    fair_model = BaselineScorer.load("models/baseline")
    results_fair = run_audit(fair_model, test_df,
                             label_col="human_score", tag="FAIR")

    # ── Audit 2: BIASED labels ────────────────────────────────────────────────
    logger.info("\nTraining biased baseline on human_score_biased …")
    biased_model = BaselineScorer()
    biased_model.fit(train_df, label_col="human_score_biased")
    biased_model.save("models/baseline_biased")
    results_biased = run_audit(biased_model, test_df,
                               label_col="human_score_biased", tag="BIASED")

    # ── Comparison plots ──────────────────────────────────────────────────────
    _plot_subgroup_metrics(
        sg_fair=results_fair["subgroup_metrics"],
        sg_biased=results_biased["subgroup_metrics"],
        save_path=FIGURES / "fairness_subgroup_metrics.png",
    )

    write_fairness_report(
        {"fair": results_fair, "biased": results_biased},
        output_path="reports/fairness_report.md",
    )

    # ── Machine-readable summary (consumed by the Streamlit dashboard) ─────────
    mit = results_biased.get("mitigation", {})
    tq  = results_biased.get("true_quality_validation", {})
    summary = {
        "pass_threshold": get("pass_threshold", 3),
        "fair": {
            "ratio":               results_fair["adverse_impact"]["ratio"],
            "adverse_impact_flag": results_fair["adverse_impact"]["adverse_impact_flag"],
            "selection_rates":     results_fair["adverse_impact"]["selection_rates"],
            "demographic_parity_difference":
                results_fair["fairlearn_metrics"]["demographic_parity_difference"],
        },
        "biased": {
            "ratio":               results_biased["adverse_impact"]["ratio"],
            "adverse_impact_flag": results_biased["adverse_impact"]["adverse_impact_flag"],
            "selection_rates":     results_biased["adverse_impact"]["selection_rates"],
            "demographic_parity_difference":
                results_biased["fairlearn_metrics"]["demographic_parity_difference"],
        },
        "mitigated": {
            "ratio":               mit.get("adverse_impact_after", {}).get("ratio"),
            "adverse_impact_flag": mit.get("adverse_impact_after", {}).get("adverse_impact_flag"),
            "selection_rates":     mit.get("adverse_impact_after", {}).get("selection_rates"),
        },
        "true_quality_validation": tq,
    }
    summary_path = Path("reports/fairness_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info(f"Fairness summary written to {summary_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("FAIRNESS AUDIT SUMMARY")
    print(f"{'='*55}")
    print(f"{'':30} {'FAIR':>10} {'BIASED':>10}")
    print(f"  {'Label adverse impact ratio':<30} "
          f"{results_fair['adverse_impact']['ratio']:>10.3f} "
          f"{results_biased['adverse_impact']['ratio']:>10.3f}")
    print(f"  {'Adverse impact flag':<30} "
          f"{'⚠️' if results_fair['adverse_impact']['adverse_impact_flag'] else '✅':>10} "
          f"{'⚠️' if results_biased['adverse_impact']['adverse_impact_flag'] else '✅':>10}")
    print(f"  {'DPD (model predictions)':<30} "
          f"{results_fair['fairlearn_metrics']['demographic_parity_difference']:>10.4f} "
          f"{results_biased['fairlearn_metrics']['demographic_parity_difference']:>10.4f}")
    if "mitigation" in results_biased:
        after = results_biased["mitigation"]["adverse_impact_after"]
        print(f"  {'After mitigation ratio':<30} {'N/A':>10} {after['ratio']:>10.3f}")
    print(f"\n✅  Reports: reports/fairness_report.md")
    print(f"✅  Figures: reports/figures/fairness_*.png")


if __name__ == "__main__":
    main()
