# 📝 JustGrade — Automated Response Scorer with Fairness Audit

> Rubric-anchored scoring of open-ended responses — **accurate** (QWK-evaluated), **fair**
> (subgroup-audited with adverse-impact testing + mitigation), and **explainable**
> (phrase-level attention highlighting). A faithful research prototype of the methodology
> behind automated interview / SJT scoring.

This project mirrors the core problem in talent assessment: predict a competency score that
agrees with expert human raters, then **prove** it's both accurate *and* fair, and show *why*
each score was given.

---

## Why this is interesting

Automated hiring scores are high-stakes, legally-exposed decisions. A model that is accurate
on average can still be **systematically unfair to a subgroup** — and, critically, the bias
often lives in the *human labels*, not the inputs. JustGrade demonstrates the full lifecycle:

1. **Accuracy** — a fine-tuned transformer scores responses, beating an embeddings baseline.
2. **Fairness** — a rigorous audit finds *nothing* on fair labels (no false alarm), *catches*
   adverse impact on biased labels, and a mitigation *fixes* it — validated against unbiased
   ground truth.
3. **Explainability** — attention-rollout phrase highlighting shows which parts of a response
   drove the score (the human-in-the-loop "Augmented Intelligence" pattern).

---

## Results

| Metric | Baseline (bge + GBM) | Transformer (DistilBERT) |
|--------|---------------------:|-------------------------:|
| **QWK** (↑)            | 0.9050 | **0.9308** |
| **QWK 95% CI**         | [0.8907, 0.9183] | **[0.9206, 0.9405]** |
| MAE (↓)                | 0.3500 | **0.2764** |
| Pearson r (↑)          | 0.9053 | **0.9308** |
| Within-1 accuracy (↑)  | 98.5%  | **100%** |

The transformer's bootstrap confidence interval barely overlaps the baseline's — the
improvement is statistically meaningful, not noise. *(Test split: 720 held-out responses.)*

### Fairness findings → [`reports/fairness_report.md`](reports/fairness_report.md)

| | Fair labels (`human_score`) | Biased labels (`human_score_biased`) | After mitigation |
|---|:---:|:---:|:---:|
| **4/5ths adverse-impact ratio** | 1.000 ✅ | **0.750 ⚠️** | 1.000 ✅ |
| Group A pass rate | 60% | 60% | 60% |
| Group B pass rate | 60% | **45%** | 60% |
| Accuracy vs *true* quality (Group B) | — | 85% | **90%** |

> **The headline arc:** the audit raises **no false alarm** on fair data, **catches** clear
> adverse impact on biased data, and the mitigation **fixes it** — while *improving* Group B's
> agreement with unbiased ground truth. Bias in the labels does not have to become bias in the
> decision.

---

## Architecture

```
 raw data ─► 1. Ingest & validate (schema contract, no-leakage check)
                │
                ▼
            2. Preprocess  (stratified train/val/test, group/question aware)
              ┌─────────────┴──────────────┐
              ▼                             ▼
   3a. BASELINE                   3b. TRANSFORMER
   bge-small embeddings           DistilBERT + regression head
   + GradientBoosting             (HF Trainer, early stopping on val QWK)
              └─────────────┬──────────────┘
                            ▼
            4. EVALUATE  QWK (+ bootstrap CI) / MAE / corr / calibration
                            ▼
            5. FAIRNESS AUDIT  subgroup metrics, 4/5ths rule,
               fairlearn parity, per-group threshold mitigation
                            ▼
            6. EXPLAINABILITY  attention-rollout phrase highlighting
                            ▼
            7. STREAMLIT APP  score live + confidence routing +
               model card + fairness dashboard
```

---

## Quick start

**Prerequisite:** unzip the dataset bundle into the project root so `data/processed/{train,val,test}.csv`
and `data/raw/responses.csv` exist (4,800 rows, pre-split).

```bash
python3 -m venv venv && source venv/bin/activate
make setup           # install dependencies
make data-check      # confirm the dataset is present + print label distributions
make all             # baseline → transformer → eval → fairness  (~15 min, CPU/MPS)
make app             # launch the Streamlit demo at http://localhost:8501
```

Or run stages individually: `make baseline`, `make transformer`, `make eval`, `make fairness`.
Run the test suite with `make test`.

> **Enterprise networks (Zscaler / corporate proxy):** model downloads use the system trust
> store via `truststore` (auto-injected in `src/__init__.py`) — no `sudo`, no cert wrangling.

---

## The Streamlit app

| Tab | What it shows |
|---|---|
| **🎯 Score a Response** | Pick one of 12 questions (auto-fills rubric + a real high/low example), score it, see the predicted score, a **confidence gauge**, a **human-review flag** for ambiguous cases, and **attention-based phrase highlights**. |
| **📊 Model Card** | Baseline vs transformer metric table (direction-aware best-cell highlighting), **bootstrap QWK confidence intervals**, confusion matrices, and **calibration curves**. |
| **⚖️ Fairness Dashboard** | Live 4/5ths ratios (fair / biased / mitigated), selection-rate before/after chart, subgroup metric comparison, and the full audit report. |

---

## Repository layout

```
src/
├── data/ingest.py           # load + validate splits (schema contract)
├── models/baseline.py       # embeddings + GBM (with disk-cached embeddings)
├── models/transformer.py    # DistilBERT regression head (batched inference)
├── models/predict.py        # unified predict(question, rubric, response)
├── eval/metrics.py          # QWK, MAE, corr, within-1, bootstrap CI
├── eval/evaluate.py         # full eval + confusion matrix + calibration + model card
├── eval/fairness.py         # subgroup metrics, 4/5ths, fairlearn, mitigation, report
├── eval/confidence.py       # prediction confidence + human-review routing
├── explain/attributions.py  # attention rollout + HTML phrase highlighting
└── utils/{config,logging}.py
app/streamlit_app.py         # 3-tab interactive demo
scripts/01–05_*.py           # phase driver scripts
tests/                       # 42 tests (metrics, attributions, confidence, config, leakage)
reports/                     # metrics.json, model_card.md, fairness_report.md, figures/
```

---

## Tech stack & rationale

| Layer | Choice | Why |
|---|---|---|
| Baseline | `BAAI/bge-small-en-v1.5` + GradientBoosting | Strong, fast, CPU-friendly — *always baseline first* |
| Main model | `distilbert-base-uncased` + regression head (HF Trainer) | Real fine-tuning, small enough for free Colab/MPS |
| Metric | Quadratic Weighted Kappa | Field standard for ordinal human-agreement |
| Fairness | `fairlearn` | Industry-standard parity/odds metrics + 4/5ths rule |
| Explainability | Attention rollout (Abnar & Zuidema, 2020) | Propagates importance through all layers; no extra deps |
| App | `streamlit` | Fastest path to a demoable interface |
| Tests | `pytest` | Engineering signal; leakage guardrail |

---

## Methodology notes (defensible choices)

- **QWK over accuracy** — penalises a 1↔5 disagreement far more than a 3↔4 one; accuracy treats
  them identically. Reported with a **bootstrap 95% CI** so the comparison is statistically honest.
- **Baseline first** — the embeddings+GBM baseline quantifies what fine-tuning actually buys
  (+0.026 QWK here), rather than reaching for a transformer reflexively.
- **Bias lives in the labels** — `group` is assigned independently of quality, so the data is
  fair by construction; the bias is injected into `human_score_biased`. This mirrors the
  real-world finding that human raters themselves can be biased — and is why auditing the
  *label distribution* (not just model predictions) is the right lens.
- **No-leakage guardrail** — `tests/test_split_no_leakage.py` asserts zero ID overlap across
  splits; it runs in CI/`make test`.
- **Confidence routing** — low-confidence predictions (raw score near a `.5` boundary) are
  flagged for human review rather than auto-scored — the Augmented-Intelligence pattern.

---

## Honest limitations

- The `group` attribute and the bias mechanism are **simulated** to demonstrate methodology,
  not real demographic data.
- Response text is **template-assembled** — fluent, but less varied than real candidate writing.
- The threshold-optimisation mitigation addresses **decision equity**, not score equity — the
  underlying scores are unchanged.
- This is a **research prototype**: no psychometric validity claim, no production SLA/auth.

---

## License & data

Uses only the provided synthetic dataset (and optionally a public set like ASAP-AES). It never
assumes access to real candidate data.
