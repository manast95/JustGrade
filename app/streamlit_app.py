"""JustGrade — Streamlit demo: score responses, view model card, explore fairness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.predict import _dummy_row
from src.utils.config import get

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JustGrade",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Cached model loaders ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading baseline model…")
def _load_baseline():
    from src.models.baseline import BaselineScorer  # noqa: PLC0415
    return BaselineScorer.load("models/baseline")


@st.cache_resource(show_spinner="Loading transformer model…")
def _load_transformer():
    from src.models.transformer import TransformerScorer  # noqa: PLC0415
    return TransformerScorer.load("models/transformer")


@st.cache_data(show_spinner=False)
def _load_question_bank() -> dict[str, dict[str, str]]:
    """Map question_id → {question, rubric, example_high, example_low} from the dataset.

    Reads the raw responses CSV (falls back to the train split) and pulls, per
    question, the prompt + rubric and one high-scoring and one low-scoring real
    response as ready-to-use examples.
    """
    for candidate in ("data/raw/responses.csv", "data/processed/train.csv"):
        path = Path(candidate)
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        return {}

    bank: dict[str, dict[str, str]] = {}
    for qid, grp in df.groupby("question_id"):
        first = grp.iloc[0]
        entry = {"question": first["question"], "rubric": first["rubric"]}
        # One high (==max) and one low (==min) human_score response as examples
        hi_rows = grp[grp["human_score"] == grp["human_score"].max()]
        lo_rows = grp[grp["human_score"] == grp["human_score"].min()]
        entry["example_high"] = str(hi_rows.iloc[0]["response"]) if len(hi_rows) else ""
        entry["example_low"]  = str(lo_rows.iloc[0]["response"]) if len(lo_rows) else ""
        bank[str(qid)] = entry
    return dict(sorted(bank.items()))


# ── Default example ───────────────────────────────────────────────────────────

DEFAULT_QUESTION = "Tell us about a time you led a group toward a goal."
DEFAULT_RUBRIC   = (
    "Score on leadership: ownership, how the candidate motivated/organized others, "
    "decisions made, and a measurable outcome. 1=poor, 5=excellent."
)
DEFAULT_HIGH = (
    "We needed to onboard three new members quickly. I took clear ownership of the "
    "situation, set a concrete goal and tracked progress openly. I kept everyone "
    "informed throughout. The group exceeded its target saving roughly 10 hours a week. "
    "It reshaped how I approach similar situations now."
)
DEFAULT_LOW = (
    "I was in a group once. We did some work together. "
    "It was okay in the end."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_colour(score: int) -> str:
    colours = {1: "#d32f2f", 2: "#f57c00", 3: "#fbc02d", 4: "#388e3c", 5: "#1976d2"}
    return colours.get(score, "#888")


def _gauge(score: int, lo: int = 1, hi: int = 5) -> str:
    filled = score - lo
    total  = hi - lo
    bars   = "█" * filled + "░" * (total - filled)
    colour = _score_colour(score)
    return (
        f'<div style="font-family:monospace;font-size:1.4rem;color:{colour};">'
        f"{bars} {score}/{hi}</div>"
    )


# ── Tab 1 — Score a response ──────────────────────────────────────────────────

def tab_score() -> None:
    lo, hi = get("label_range", [1, 5])
    bank = _load_question_bank()

    col_in, col_out = st.columns([3, 2], gap="large")

    with col_in:
        st.subheader("Input")

        # ── Question picker (prefills question + rubric from the dataset) ──────
        options = ["✏️ Custom"] + [
            f"{qid} — {bank[qid]['question'][:55]}…" for qid in bank
        ]
        choice = st.selectbox("Pick a question (or write your own)", options)

        if choice == "✏️ Custom":
            q_default, r_default, ex_high, ex_low = (
                DEFAULT_QUESTION, DEFAULT_RUBRIC, DEFAULT_HIGH, DEFAULT_LOW
            )
        else:
            qid = choice.split(" — ")[0]
            entry = bank[qid]
            q_default = entry["question"]
            r_default = entry["rubric"]
            ex_high = entry.get("example_high") or DEFAULT_HIGH
            ex_low  = entry.get("example_low") or DEFAULT_LOW

        # Example loader: lets the user drop in a real high/low response
        ex_pick = st.radio(
            "Example response to load", ["High-scoring", "Low-scoring"],
            horizontal=True, key=f"ex_{choice}",
        )
        resp_default = ex_high if ex_pick == "High-scoring" else ex_low

        question = st.text_area("Question / Prompt", value=q_default, height=80,
                                key=f"q_{choice}")
        rubric   = st.text_area("Scoring Rubric", value=r_default, height=80,
                                key=f"r_{choice}")
        response = st.text_area("Candidate Response", value=resp_default, height=140,
                                key=f"resp_{choice}_{ex_pick}")

        model_choice = st.radio(
            "Model", ["Transformer (DistilBERT)", "Baseline (Embeddings + GBM)"],
            horizontal=True,
        )
        use_transformer = model_choice.startswith("Transformer")
        run = st.button("Score response", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Result")
        if not run:
            st.info("Pick a question, load an example (or edit freely), and click "
                    "**Score response**.")
            return

        with st.spinner("Scoring…"):
            df = _dummy_row(question, rubric, response)
            try:
                if use_transformer:
                    model = _load_transformer()
                else:
                    model = _load_baseline()
                raw   = float(model.predict_raw(df)[0])
                score = max(lo, min(hi, round(raw)))
            except Exception as exc:
                st.error(f"Model error: {exc}")
                return

        # ── Confidence + human-review routing ─────────────────────────────────
        from src.eval.confidence import (  # noqa: PLC0415
            confidence_band, prediction_confidence,
        )
        conf = float(prediction_confidence([raw])[0])
        min_conf = get("review.min_confidence", 0.5)
        band = confidence_band(conf)

        st.markdown(_gauge(score, lo, hi), unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Predicted score", f"{score} / {hi}",
                  delta=f"raw={raw:.3f}", delta_color="off")
        m2.metric("Confidence", f"{conf:.0%}", delta=band, delta_color="off")

        if conf < min_conf:
            st.warning(
                f"⚠️ **Low confidence ({conf:.0%})** — the raw score {raw:.2f} sits near "
                f"a decision boundary. In a production setting this case would be "
                f"**routed to a human rater** (Augmented-Intelligence pattern) rather "
                f"than auto-scored."
            )
        else:
            st.success(f"✅ Confident automated score ({band.lower()}).")

        st.divider()

        if use_transformer:
            with st.spinner("Computing phrase highlights (attention rollout)…"):
                try:
                    from src.explain.attributions import (  # noqa: PLC0415
                        response_attributions, to_html_highlight,
                    )
                    pairs = response_attributions(model, question, rubric, response)
                    html  = to_html_highlight(pairs, top_k=10)
                    st.subheader("Phrase importance")
                    st.caption(
                        "Darker = higher attention weight (attention rollout across "
                        "all 6 DistilBERT layers). Shows *why* the model gave this score."
                    )
                    st.markdown(html, unsafe_allow_html=True)

                    # Top-5 word table
                    top5 = sorted(pairs, key=lambda x: x[1], reverse=True)[:5]
                    st.dataframe(
                        pd.DataFrame(top5, columns=["Word", "Importance"]).assign(
                            Importance=lambda d: d["Importance"].map("{:.3f}".format)
                        ),
                        hide_index=True, use_container_width=True,
                    )
                except Exception as exc:
                    st.warning(f"Attributions unavailable: {exc}")
        else:
            st.info(
                "Phrase highlighting is available for the Transformer model only. "
                "Switch to *Transformer* to see attention-based explanations."
            )


# ── Tab 2 — Model card ────────────────────────────────────────────────────────

def tab_model_card() -> None:
    metrics_path = Path("reports/metrics.json")
    card_path    = Path("reports/model_card.md")

    if not metrics_path.exists():
        st.warning("Run `make eval` first to generate metrics.")
        return

    data = json.loads(metrics_path.read_text())

    st.subheader("Performance comparison")
    metric_labels = {
        "qwk":           "QWK (↑ better)",
        "mae":           "MAE (↓ better)",
        "pearson":       "Pearson r (↑)",
        "spearman":      "Spearman ρ (↑)",
        "within_one_acc": "Within-1 acc (↑)",
    }
    rows = []
    for mk, label in metric_labels.items():
        row = {"Metric": label}
        for model_name, mdata in data.items():
            row[model_name.title()] = round(mdata.get(mk, float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Metric")

    def _highlight_best(row: pd.Series) -> list[str]:
        """Green-highlight the best model per metric, honouring metric direction.

        MAE is lower-is-better (label contains '↓'); everything else is
        higher-is-better. A flat highlight_max would wrongly green the worst MAE.
        """
        if row.isna().all():
            return ["" for _ in row]
        best = row.min() if "↓" in str(row.name) else row.max()
        return ["background-color: #c8e6c9" if v == best else "" for v in row]

    st.dataframe(
        df.style.apply(_highlight_best, axis=1).format("{:.4f}"),
        use_container_width=True,
    )

    st.caption(
        "**QWK** (Quadratic Weighted Kappa) is the primary metric — "
        "field standard for ordinal scoring, penalises large disagreements quadratically. "
        f"Transformer gains +{data.get('transformer',{}).get('qwk',0) - data.get('baseline',{}).get('qwk',0):.3f} QWK "
        "over the embedding+GBM baseline."
    )

    # QWK bootstrap confidence intervals
    ci_bits = []
    for model_name, md in data.items():
        if "qwk_ci_low" in md and "qwk_ci_high" in md:
            level = int(md.get("ci_level", 0.95) * 100)
            ci_bits.append(
                f"**{model_name.title()}**: QWK {md['qwk']:.3f} "
                f"({level}% CI [{md['qwk_ci_low']:.3f}, {md['qwk_ci_high']:.3f}])"
            )
    if ci_bits:
        st.info("📐 Bootstrap confidence intervals — " + "  •  ".join(ci_bits))

    st.divider()
    st.subheader("Confusion matrices")
    c1, c2 = st.columns(2)
    for col, name, path in [
        (c1, "Baseline",    "reports/figures/confusion_matrix_baseline.png"),
        (c2, "Transformer", "reports/figures/confusion_matrix_transformer.png"),
    ]:
        p = Path(path)
        if p.exists():
            col.image(str(p), caption=name, use_container_width=True)
        else:
            col.info(f"Run `make eval` to generate the {name} confusion matrix.")

    st.divider()
    st.subheader("Calibration curves")
    st.caption(
        "When the model predicts a score *s*, what is the mean true score of those "
        "responses? Points on the diagonal = well-calibrated."
    )
    c3, c4 = st.columns(2)
    for col, name, path in [
        (c3, "Baseline",    "reports/figures/calibration_baseline.png"),
        (c4, "Transformer", "reports/figures/calibration_transformer.png"),
    ]:
        p = Path(path)
        if p.exists():
            col.image(str(p), caption=name, use_container_width=True)
        else:
            col.info(f"Run `make eval` to generate the {name} calibration curve.")

    if card_path.exists():
        st.divider()
        with st.expander("Full model card (markdown)", expanded=False):
            st.markdown(card_path.read_text())


# ── Tab 3 — Fairness dashboard ────────────────────────────────────────────────

def tab_fairness() -> None:
    report_path  = Path("reports/fairness_report.md")
    summary_path = Path("reports/fairness_summary.json")

    st.subheader("Fairness audit summary")

    # Read live results from the audit; fall back gracefully if not yet run.
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            summary = None

    if summary is None:
        st.warning("Run `make fairness` to generate the audit results.")
        threshold = 3
    else:
        threshold = summary.get("pass_threshold", 3)

        def _fmt(val) -> str:
            return f"{val:.3f}" if isinstance(val, (int, float)) else "N/A"

        def _flag(flagged) -> str:
            if flagged is None:
                return "Not run"
            return "Adverse impact ⚠️" if flagged else "No adverse impact ✅"

        fair = summary["fair"]
        biased = summary["biased"]
        mitig = summary["mitigated"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Fair labels — 4/5ths ratio", _fmt(fair["ratio"]),
                    _flag(fair["adverse_impact_flag"]), delta_color="off")
        col2.metric("Biased labels — 4/5ths ratio", _fmt(biased["ratio"]),
                    _flag(biased["adverse_impact_flag"]), delta_color="inverse")
        col3.metric("After mitigation — 4/5ths ratio", _fmt(mitig.get("ratio")),
                    _flag(mitig.get("adverse_impact_flag")), delta_color="off")

    st.caption(
        "The **4/5ths (80%) rule**: if the selection rate of the disadvantaged group "
        "is less than 80% of the advantaged group's rate, adverse impact is flagged. "
        f"Threshold used: score ≥ {threshold} = 'pass'."
    )

    st.divider()
    st.subheader("Selection rates before & after mitigation")
    fig_path = Path("reports/figures/fairness_selection_rates_biased.png")
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.info("Run `make fairness` to generate figures.")

    st.divider()
    st.subheader("Subgroup metrics — fair vs biased labels")
    sg_path = Path("reports/figures/fairness_subgroup_metrics.png")
    if sg_path.exists():
        st.image(str(sg_path), use_container_width=True)

    st.divider()
    st.subheader("Design")
    st.markdown("""
| | Fair audit (`human_score`) | Biased audit (`human_score_biased`) |
|---|---|---|
| **Label adverse impact ratio** | 1.000 ✅ | 0.750 ⚠️ |
| **Group A pass rate** | 60% | 60% |
| **Group B pass rate** | 60% | 45% |
| **After mitigation** | — | 60% ✅ |
| **Accuracy vs true quality — Group B** | — | 85% → 90% |

**Key insight:** the `group` attribute is independent of response quality by construction —
bias lives in the *labels*, not the inputs. This mirrors SHL's published finding that
human rater bias is a primary source of assessment unfairness.

**Mitigation method:** per-group threshold optimisation on continuous model scores.
No retraining required; post-processing only.
""")

    if report_path.exists():
        st.divider()
        with st.expander("Full fairness report (markdown)", expanded=False):
            st.markdown(report_path.read_text())
    else:
        st.info("Run `make fairness` to generate the full report.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📝 JustGrade")
    st.caption(
        "Automated Response Scorer with Fairness Audit — "
        "rubric-anchored scoring, QWK-evaluated, bias-audited, phrase-explainable."
    )

    tab1, tab2, tab3 = st.tabs([
        "🎯 Score a Response",
        "📊 Model Card",
        "⚖️ Fairness Dashboard",
    ])
    with tab1:
        tab_score()
    with tab2:
        tab_model_card()
    with tab3:
        tab_fairness()


if __name__ == "__main__":
    main()
