# 📝 JustGrade — Automated Response Scorer with Fairness Audit

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-42%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-manast95%2FJustGrade-black?logo=github)](https://github.com/manast95/JustGrade)

> Rubric-anchored scoring of open-ended responses — **accurate** (QWK 0.931), **fair**
> (4/5ths adverse-impact audited + mitigated), and **explainable** (phrase-level attention
> highlighting). A research prototype of the methodology behind automated interview / SJT scoring.

This project mirrors the core problem in talent assessment: predict a competency score that
agrees with expert human raters, then **prove** it is both accurate *and* fair, and show *why*
each score was given.

---

## Why this matters

Automated hiring scores are high-stakes, legally-exposed decisions. A model that is accurate
on average can still be **systematically unfair to a subgroup** — and critically, the bias
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
| **QWK** (↑)            | 0.905  | **0.931**  |
| **QWK 95% CI**         | [0.891, 0.918] | **[0.921, 0.941]** |
| MAE (↓)                | 0.350  | **0.276**  |
| Pearson r (↑)          | 0.905  | **0.931**  |
| Within-1 accuracy (↑)  | 98.5%  | **100%**   |

> The transformer's CI lower bound (0.921) exceeds the baseline's upper bound (0.918) —
> the improvement is statistically meaningful. *(720 held-out test responses, 1 000-resample bootstrap.)*

### Fairness audit — [`reports/fairness_report.md`](reports/fairness_report.md)

| | Fair labels | Biased labels | After mitigation |
|---|:---:|:---:|:---:|
| **4/5ths adverse-impact ratio** | 1.000 ✅ | **0.750 ⚠️** | 1.000 ✅ |
| Group A pass rate | 60% | 60% | 60% |
| Group B pass rate | 60% | **45%** | 60% |
| Group B accuracy vs true quality | — | 85% | **90%** |

> **The headline arc:** no false alarm on fair data → catches adverse impact on biased data →
> mitigation resolves it while *improving* Group B's alignment with unbiased ground truth.

---

## Screenshots

| Score a Response | Model Card | Fairness Dashboard |
|:---:|:---:|:---:|
| Predicted score + confidence + phrase highlights | Bootstrap CI + confusion matrix + calibration | 4/5ths ratio + selection-rate chart + full report |

---

## Architecture

```
 raw data ─► 1. Ingest & validate (schema contract, no-leakage check)
                │
                ▼
            2. Preprocess  (stratified train/val/test, group/question-aware)
              ┌─────────────┴──────────────┐
              ▼                             ▼
   3a. BASELINE                   3b. TRANSFORMER
   bge-small embeddings           DistilBERT + regression head
   + GradientBoosting             (HF Trainer, early stopping on val QWK)
              └─────────────┬──────────────┘
                            ▼
            4. EVALUATE  QWK + bootstrap CI / MAE / calibration curve
                            ▼
            5. FAIRNESS AUDIT  subgroup metrics, 4/5ths rule,
               fairlearn parity/odds, per-group threshold mitigation
                            ▼
            6. EXPLAINABILITY  attention rollout → phrase highlighting
                            ▼
            7. STREAMLIT APP  score live + confidence routing +
               model card + fairness dashboard
```

---

## Getting started

### 1. Clone & install

```bash
git clone https://github.com/manast95/JustGrade.git
cd JustGrade
python3 -m venv venv && source venv/bin/activate
make setup
```

### 2. Add the dataset

The data CSVs are not included in the repo (they contain synthetic candidate data).
Place the dataset bundle so the following files exist:

```
data/
├── raw/responses.csv          # 4 800 rows
└── processed/
    ├── train.csv              # 3 360 rows (70 %)
    ├── val.csv                #   720 rows (15 %)
    └── test.csv               #   720 rows (15 %)
```

Verify the data is present:
```bash
make data-check
```

### 3. Train & evaluate

```bash
make all      # baseline → transformer → eval → fairness audit  (~15 min on CPU/MPS)
make app      # launch the Streamlit demo at http://localhost:8501
```

Or run stages individually:

```bash
make baseline      # train bge + GBM baseline
make transformer   # fine-tune DistilBERT (~13 min on M1)
make eval          # metrics, confusion matrix, calibration, model card
make fairness      # full fairness audit + fairness_report.md
make test          # 42 pytest tests with coverage
```

> **Enterprise / Zscaler networks:** SSL inspection is handled automatically via `truststore`
> (injected in `src/__init__.py`) — no `sudo`, no manual cert setup needed.

---

## The Streamlit app

| Tab | What it shows |
|---|---|
| **🎯 Score a Response** | Pick one of 12 competency questions (auto-fills rubric + a real high/low response from the dataset), score it with either model, see the **predicted score**, **confidence gauge**, a **human-review flag** for ambiguous cases, and **attention-based phrase highlights** |
| **📊 Model Card** | Baseline vs transformer metric table (direction-aware highlighting), **bootstrap QWK confidence intervals**, confusion matrices, and **calibration curves** |
| **⚖️ Fairness Dashboard** | Live 4/5ths ratios (fair / biased / mitigated), selection-rate before/after chart, subgroup QWK/MAE comparison, and the full audit report |

---

## Repository layout

```
JustGrade/
├── README.md
├── SPEC.md                          # full product + engineering spec
├── config.yaml                      # all hyperparams, paths, thresholds
├── Makefile                         # setup / baseline / transformer / eval / fairness / app / test
├── requirements.txt
├── data/
│   ├── generate_synthetic.py        # reproduces the dataset (seed=42)
│   └── README.md                    # schema, generation method, fairness mechanism
├── src/
│   ├── data/ingest.py               # load + validate splits (schema contract)
│   ├── models/
│   │   ├── baseline.py              # embeddings + GBM (disk-cached embeddings)
│   │   ├── transformer.py           # DistilBERT regression head (batched inference)
│   │   └── predict.py              # unified predict(question, rubric, response)
│   ├── eval/
│   │   ├── metrics.py               # QWK, MAE, corr, within-1, bootstrap CI
│   │   ├── evaluate.py              # eval + confusion matrix + calibration + model card
│   │   ├── fairness.py              # subgroup metrics, 4/5ths, mitigation, report
│   │   └── confidence.py            # prediction confidence + human-review routing
│   └── explain/attributions.py      # attention rollout + HTML phrase highlighting
├── app/streamlit_app.py             # 3-tab interactive demo
├── scripts/01–05_*.py               # phase driver scripts
├── tests/                           # 42 tests (metrics, attributions, confidence, config, leakage)
└── reports/                         # metrics.json, model_card.md, fairness_report.md, figures/
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Baseline | `BAAI/bge-small-en-v1.5` + GradientBoosting | Fast, CPU-friendly; establishes what fine-tuning buys |
| Main model | `distilbert-base-uncased` + regression head | Real fine-tuning skill; small enough for free Colab / M1 |
| Metric | Quadratic Weighted Kappa | Field standard for ordinal human-agreement scoring |
| Fairness | `fairlearn` | Industry-standard demographic parity / equalized odds |
| Explainability | Attention rollout (Abnar & Zuidema, 2020) | Propagates importance through all layers; no extra deps |
| App | `streamlit` | Fastest path to a demoable interface |
| Tests | `pytest` + `pytest-cov` | Engineering discipline; leakage guardrail |

---

## Methodology notes

- **QWK over accuracy** — penalises a 1↔5 disagreement far more than a 3↔4 one; accuracy treats them identically. Reported with a bootstrap 95% CI so the comparison is statistically honest.
- **Baseline first** — the embeddings + GBM baseline quantifies what fine-tuning actually buys (+0.026 QWK), rather than reaching for a transformer reflexively.
- **Bias in the labels** — `group` is assigned independently of response quality; bias is injected into `human_score_biased`. This mirrors the real-world finding that human rater bias is the primary source of assessment unfairness — and is why auditing the *label distribution* (not just model predictions) is the right lens.
- **No-leakage guardrail** — `tests/test_split_no_leakage.py` asserts zero ID overlap across splits and runs in every `make test` invocation.
- **Confidence routing** — low-confidence predictions (raw score near a `.5` boundary) are flagged for human review rather than auto-scored — the Augmented-Intelligence pattern.

---

## Honest limitations

- The `group` attribute and bias mechanism are **simulated** to demonstrate methodology, not real demographic groups.
- Response text is **template-assembled** — fluent, but less varied than real candidate writing.
- The threshold-optimisation mitigation addresses **decision equity** (who passes), not score equity — underlying scores are unchanged.
- This is a **research prototype**: no psychometric validity claim, no production SLA or auth.

---

## License

MIT — see [LICENSE](LICENSE).
