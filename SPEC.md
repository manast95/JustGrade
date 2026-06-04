# PROJECT 1 — Automated Response Scorer with Fairness Audit
## Complete Build Specification (Claude Code–ready)

> **What this is:** A full, exhaustive product + engineering spec for an NLP system that scores open-ended text responses against a rubric (like SHL's automated interview/SJT scoring), evaluated for accuracy AND audited for demographic fairness, with explainability. Hand this whole document to Claude Code and build it phase by phase.
>
> **Why it wins the SHL interview:** This *is* SHL's core problem. It closes your two gaps at once (NLP/LLMs + fairness), and gives you an authentic, end-to-end story: "I built a rubric-anchored response scorer, measured it with the field-standard metric, audited it for subgroup bias, and added phrase-level explainability."

---

## 0. HOW TO USE THIS DOC WITH CLAUDE CODE

1. Create an empty folder, open Claude Code in it.
2. Paste the **Master Kickoff Prompt** (Section 13) first.
3. Then build **phase by phase** using the phase prompts in Section 8 — don't ask for the whole app in one shot; review each phase, run it, then proceed.
4. Keep this doc in the repo as `SPEC.md` so Claude Code can reference it (`Read SPEC.md`).

---

## 1. PROJECT OVERVIEW

**Goal.** Given (a) a question/prompt, (b) a scoring rubric, and (c) a candidate's free-text response, predict a competency score (e.g., 1–5) that agrees with expert human raters — then prove it's accurate and fair.

**Three pillars (each is a deliverable and an interview talking point):**
1. **Accuracy** — predict scores that correlate with human ratings (metric: Quadratic Weighted Kappa).
2. **Fairness** — audit score/error distributions across subgroups; run an adverse-impact check; apply one mitigation.
3. **Explainability** — surface *why* a response got its score (phrase highlighting + feature attributions).

**Non-goals (state these to avoid scope creep):** not building a production service with auth/SLA; not collecting real candidate data; not claiming psychometric validity — it's a faithful research prototype of the *methodology*.

---

## 2. DATASET STRATEGY

**✅ The synthetic dataset is ALREADY BUILT** and ships with this spec as `response_scorer_dataset.zip`. Unzip it into the project root so you get:

```
data/
├── generate_synthetic.py        # the generator (reproducible; re-run to scale/tune)
├── raw/responses.csv            # full dataset — 4,800 rows
├── processed/train.csv          # 3,360 rows (70%)
├── processed/val.csv            #   720 rows (15%)
├── processed/test.csv           #   720 rows (15%)
└── README.md                    # schema, generation method, fairness mechanism, caveats
```

You do **not** need to generate data to get started — drop the files in and train. (The generator stays in the tree so you can scale via `--per_cell`.)

**Schema of the provided data (every row):**
| Column | Meaning |
|---|---|
| `id` | unique row id |
| `question_id` | one of 12 competency questions (`Q01`…`Q12`) |
| `question` | the prompt text |
| `rubric` | scoring criteria for that question |
| `group` | simulated sensitive attribute **A**/**B** (assigned independent of quality) |
| `true_quality` | latent intended quality 1–5 (synthetic-only ground truth) |
| `human_score` | **FAIR** observed label (== `true_quality`) — **default training/eval label** |
| `human_score_biased` | **BIASED** observed label (group B systematically under-scored) |
| `response` | the free-text answer |

**Why two label columns (this is the heart of the fairness demo):** `group` is assigned independently of quality, so the data is *fair by construction*. The bias lives in the **labels**, mirroring SHL's published finding that human ground truth itself can be biased.
- Train/audit on `human_score` (fair) → groups have identical score distributions → **4/5ths ratio ≈ 1.00** (no adverse impact). This is the audit's *null case*: a good audit finds **nothing**.
- Train/audit on `human_score_biased` → group B disadvantaged → **4/5ths ratio ≈ 0.77 (< 0.80 → adverse impact)**. This is what the audit should **catch** and the mitigation should **fix**, validated against `true_quality`.

**Validated learnable signal:** response text quality genuinely scales with the score (length, STAR completeness, specificity, competency vocabulary). A trivial TF-IDF + logistic baseline already hits **QWK ≈ 0.92** on the fair labels, confirming the signal is learnable with headroom for a fine-tuned transformer.

**Optional — pair with a real dataset for the accuracy headline.** This synthetic set's main job is the *fairness audit* (it carries the `group` attribute). For more credible accuracy numbers, you can additionally run the accuracy model on a real public set:
- **ASAP-AES / ASAP-SAS** (Automated Student Assessment Prize) — free, real essays + human scores (needs a free Kaggle account; normalize scoring scales). Real data is harder, so a 0.78 there reads as more credible than 0.95 on synthetic. Treat this as a nice-to-have, not required.

**Honest caveats (state these in your README too):** the text is template-assembled (fluent but less varied than real responses); the `group` attribute and the bias are simulated to demonstrate *method*, not real groups.

---

## 3. TECH STACK (with rationale — useful as interview answers)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Standard for ML |
| Baseline model | `sentence-transformers` (e.g., `BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`) embeddings + scikit-learn regressor (Ridge / GradientBoosting) | Strong, fast, CPU-friendly baseline; "always baseline first" |
| Main model | Fine-tuned transformer regression head (`distilbert-base-uncased` → 1 output) via HuggingFace `Trainer` | Shows real fine-tuning skill; small enough for free Colab/GPU |
| Optional PEFT | LoRA via `peft` | Demonstrates parameter-efficient fine-tuning (name-drops well) |
| Metrics | QWK (primary), MAE, Pearson/Spearman, accuracy-within-1 | QWK is the field standard for ordinal scoring |
| Fairness | `fairlearn` (and/or `aif360`) | Industry-standard; instant SHL credibility |
| Explainability | attention/`captum` or `shap` + token-level highlighting | Mirrors SHL's "phrase highlighting" |
| Experiment tracking | simple CSV/JSON logs (or `mlflow` stretch) | Reproducibility |
| App/UI | `streamlit` | Fastest way to a demoable interface |
| Packaging | `pip` + `requirements.txt` + `Makefile` | Clean, reproducible |
| Tests | `pytest` | Engineering signal |

**Embedding model note (2026):** good open-source picks are `BAAI/bge-small-en-v1.5` (light) or `BAAI/bge-m3` (stronger, multilingual). If you prefer an API, OpenAI `text-embedding-3-large` or Voyage are top-tier — but local keeps it free and offline.

---

## 4. ARCHITECTURE

```
                ┌─────────────────────────────────────────────┐
   raw data ──► │ 1. Ingest & validate (data contract)        │
                └───────────────┬─────────────────────────────┘
                                ▼
                ┌─────────────────────────────────────────────┐
                │ 2. Preprocess (clean, split by GROUP/QUESTION │
                │    to avoid leakage; train/val/test)          │
                └───────────────┬─────────────────────────────┘
                  ┌─────────────┴──────────────┐
                  ▼                             ▼
        ┌───────────────────┐        ┌────────────────────────┐
        │ 3a. BASELINE       │        │ 3b. TRANSFORMER         │
        │ embeddings + Ridge │        │ fine-tuned regressor    │
        │ /GradientBoosting  │        │ (DistilBERT + head)     │
        └─────────┬─────────┘        └───────────┬────────────┘
                  └─────────────┬────────────────┘
                                ▼
                ┌─────────────────────────────────────────────┐
                │ 4. EVALUATE: QWK / MAE / corr (overall)       │
                └───────────────┬─────────────────────────────┘
                                ▼
                ┌─────────────────────────────────────────────┐
                │ 5. FAIRNESS AUDIT: metrics by group,          │
                │    adverse impact (4/5ths), mitigation        │
                └───────────────┬─────────────────────────────┘
                                ▼
                ┌─────────────────────────────────────────────┐
                │ 6. EXPLAINABILITY: token/phrase attributions  │
                └───────────────┬─────────────────────────────┘
                                ▼
                ┌─────────────────────────────────────────────┐
                │ 7. STREAMLIT APP: score a response live +     │
                │    show highlights + show fairness dashboard  │
                └─────────────────────────────────────────────┘
```

---

## 5. REPOSITORY STRUCTURE

```
response-scorer/
├── README.md
├── SPEC.md                      # this document
├── requirements.txt
├── Makefile
├── config.yaml                  # paths, model names, hyperparams, label range
├── data/                       # ✅ PRE-BUILT — unzip response_scorer_dataset.zip here
│   ├── generate_synthetic.py    # generator (reproducible; --per_cell to scale)
│   ├── raw/responses.csv        # full dataset (4,800 rows)
│   ├── processed/               # train.csv / val.csv / test.csv (already split)
│   └── README.md                # schema + fairness mechanism + caveats
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── ingest.py            # load data/processed/*.csv + validate schema
│   │   ├── (data already generated — see data/generate_synthetic.py at data root)
│   │   └── split.py             # OPTIONAL re-split (data ships pre-split); group/question-aware
│   ├── models/
│   │   ├── baseline.py          # embeddings + sklearn regressor
│   │   ├── transformer.py       # HF fine-tuned regression head (+ optional LoRA)
│   │   └── predict.py           # unified predict(question, rubric, response)->score
│   ├── eval/
│   │   ├── metrics.py           # qwk, mae, pearson, within_1
│   │   ├── evaluate.py          # run full eval, write reports
│   │   └── fairness.py          # subgroup metrics, adverse impact, mitigation
│   ├── explain/
│   │   └── attributions.py      # token/phrase importance + HTML highlight
│   └── utils/
│       ├── config.py            # load config.yaml
│       └── logging.py
├── app/
│   └── streamlit_app.py         # interactive demo + fairness dashboard
├── notebooks/
│   └── 01_exploration.ipynb     # optional EDA
├── reports/
│   ├── metrics.json
│   ├── fairness_report.md
│   └── figures/                 # confusion matrix, score dist by group, etc.
├── tests/
│   ├── test_metrics.py
│   ├── test_split_no_leakage.py
│   └── test_predict.py
└── scripts/
    ├── 01_prepare_data.py
    ├── 02_train_baseline.py
    ├── 03_train_transformer.py
    ├── 04_evaluate.py
    └── 05_fairness_audit.py
```

---

## 6. DETAILED COMPONENT SPECS

### 6.1 `src/data/ingest.py`
- `load_split(name) -> pd.DataFrame`: read `data/processed/{train,val,test}.csv`; **validate** columns against the Section 2 schema; raise on missing fields; log row counts and label distribution.
- `validate_schema(df) -> None`: assert required columns exist (`id, question_id, question, rubric, group, true_quality, human_score, human_score_biased, response`), label within configured range, no nulls in required fields.
- `get_label(df, label_col) -> Series`: return the chosen label column (`human_score` for fair runs, `human_score_biased` for the bias demo) — driven by `config.yaml`.

### 6.2 `data/generate_synthetic.py` (PROVIDED — runs at the data root, not in src)
The dataset is already generated; this script is included only for reproducibility/scaling.
- `python data/generate_synthetic.py --per_cell 40` → 4,800 rows (default). `--per_cell 80` → 9,600 rows.
- Internals worth knowing for interview: response text quality scales with score (length, STAR completeness, specificity); `group` is assigned independent of quality; `human_score`=fair label, `human_score_biased`=group-B-under-scored label; `maybe_blur(blur=...)` controls realistic label noise, `biased_label(p_down=...)` controls bias strength. Seeded (42) for reproducibility.
- **Do not regenerate unless you intend to** — the shipped split is already verified (0 ID overlap across train/val/test; bias mechanism produces 4/5ths ≈ 0.77).

### 6.3 `src/data/split.py` (OPTIONAL — data already ships pre-split)
The bundle includes `train/val/test.csv` already split (stratified on question×score×group, 0 ID overlap — verified). Implement this only if you want to re-split.
- `make_splits(df, by="question", ratios=(0.7,0.15,0.15), seed=42)`: supports the stricter **split-by-question** setup (no question shared across splits) for a harder generalization test. The shipped default stratifies so all questions appear in all splits (task = "score a response to a *known* question against its rubric"); both are defensible — know the difference, it's a Part-11 leakage talking point.
- If you re-split by question, add a `test_split_no_leakage.py` asserting no `question_id` appears in two splits.

### 6.4 `src/models/baseline.py`
- `class BaselineScorer`:
  - `fit(df)`: build input text = f"Q: {question}\nRubric: {rubric}\nResponse: {response}" → embed with sentence-transformers → fit Ridge or GradientBoostingRegressor.
  - `predict(df) -> np.ndarray`
  - `save/load`
- Log baseline QWK/MAE to `reports/metrics.json` under key `baseline`.

### 6.5 `src/models/transformer.py`
- `class TransformerScorer`:
  - Tokenize concatenated (question + rubric + response) with truncation.
  - `distilbert-base-uncased` backbone + a regression head (single linear unit); loss = MSE (or ordinal/`SmoothL1`).
  - Train via HF `Trainer` with early stopping on val QWK.
  - **Optional LoRA**: wrap with `peft` `LoraConfig` (target attention modules) → log that ~<1% params train.
  - Round/clip predictions to the integer label range for QWK.
  - `save/load`; log under key `transformer`.

### 6.6 `src/eval/metrics.py`
- `quadratic_weighted_kappa(y_true, y_pred) -> float`
- `mae`, `pearson`, `spearman`, `within_one_accuracy`
- Pure functions, fully unit-tested.

### 6.7 `src/eval/evaluate.py`
- `evaluate_model(model, test_df) -> dict`: compute all metrics; save confusion matrix figure; write `reports/metrics.json`.

### 6.8 `src/eval/fairness.py` (the differentiator — make it shine)
- `subgroup_metrics(df, y_pred, group_col) -> pd.DataFrame`: per-group QWK, MAE, mean predicted score, "pass rate" at a chosen threshold.
- `adverse_impact(df, y_pred, group_col, threshold) -> dict`: compute selection rate per group at a pass threshold; report the **4/5ths (80%) rule** ratio and a pass/fail flag.
- `fairness_metrics_fairlearn(...)`: demographic parity difference, equal-opportunity difference, equalized-odds difference using `fairlearn.metrics`.
- `mitigate(...)`: implement **one** mitigation and show before/after:
  - Pre-processing: reweighing OR
  - Post-processing: per-group threshold adjustment / `fairlearn` `ThresholdOptimizer`.
- `write_fairness_report(...) -> reports/fairness_report.md`: a clean markdown with tables + the headline finding + the mitigation result. **This file is a portfolio piece by itself.**

### 6.9 `src/explain/attributions.py`
- `token_attributions(model, text) -> List[(token, weight)]`: via attention rollout or `captum` Integrated Gradients.
- `to_html_highlight(tokens, weights) -> str`: render the response with important phrases highlighted (mirrors SHL's phrase highlighting).

### 6.10 `app/streamlit_app.py`
- **Tab 1 — Score a response:** text boxes for question/rubric/response → predicted score + confidence + highlighted phrases.
- **Tab 2 — Model card:** metrics table (baseline vs transformer), confusion matrix.
- **Tab 3 — Fairness dashboard:** per-group metrics, adverse-impact ratio, before/after mitigation chart.

---

## 7. CONFIG (`config.yaml`)

```yaml
label_range: [1, 5]
pass_threshold: 3            # for adverse-impact selection-rate analysis
seed: 42
data:
  train_path: data/processed/train.csv
  val_path: data/processed/val.csv
  test_path: data/processed/test.csv
  label_col: human_score          # FAIR labels by default; switch to human_score_biased for the bias demo
baseline:
  embedding_model: BAAI/bge-small-en-v1.5
  regressor: gradient_boosting   # or ridge
transformer:
  backbone: distilbert-base-uncased
  max_length: 384
  lr: 2.0e-5
  epochs: 4
  batch_size: 16
  use_lora: false
fairness:
  group_col: group
  reference_col: true_quality     # unbiased ground truth to validate mitigation against
  mitigation: threshold_optimizer   # or reweighing
```

---

## 8. PHASED BUILD PLAN (these are your Claude Code prompts)

Run these one at a time. After each, run the code, confirm it works, commit, then continue.

**Phase 0 — Scaffold + confirm data.**
> "Read SPEC.md. First confirm the pre-built dataset is present: `data/processed/train.csv`, `val.csv`, `test.csv` and `data/raw/responses.csv` (from response_scorer_dataset.zip). If missing, stop and tell me to unzip the bundle. Then create the repo structure in Section 5, a `requirements.txt` with the Section 3 stack, a `Makefile` (`setup`, `baseline`, `transformer`, `eval`, `fairness`, `app`, `test`), `config.yaml` from Section 7, and `src/utils/config.py` + `logging.py`. Stub all modules with docstrings and signatures from Section 6. Don't implement logic yet."

**Phase 1 — Data loading (no generation — data already exists).**
> "Implement `src/data/ingest.py` (load `data/processed/*.csv`, validate the Section 2 schema, and a `get_label` that returns the column named by `config.data.label_col`). The data is already generated and split — do NOT write a generator and do NOT regenerate. `src/data/split.py` is optional (only if I later ask to re-split by question). Add a tiny smoke test that loads each split and prints row counts + label distribution so I can confirm end-to-end immediately."

**Phase 2 — Baseline.**
> "Implement `src/models/baseline.py` (sentence-transformers embeddings + regressor), `src/eval/metrics.py` (QWK + others, fully unit-tested in `tests/test_metrics.py`), and `scripts/02_train_baseline.py`. Print and save baseline metrics to reports/metrics.json."

**Phase 3 — Transformer.**
> "Implement `src/models/transformer.py` (DistilBERT regression head via HF Trainer, early stopping on val QWK, optional LoRA controlled by config), `src/models/predict.py` (unified predict), and `scripts/03_train_transformer.py`. Compare against baseline in reports/metrics.json."

**Phase 4 — Evaluation.**
> "Implement `src/eval/evaluate.py`: full metrics, confusion-matrix figure, a model-card table comparing baseline vs transformer. Add `scripts/04_evaluate.py`."

**Phase 5 — Fairness (spend extra care here — this is the differentiator).**
> "Implement `src/eval/fairness.py` per Section 6.8: subgroup metrics, adverse-impact 4/5ths analysis, fairlearn fairness metrics, ONE mitigation with before/after, and a generated `reports/fairness_report.md` with tables and a written headline finding. Add `scripts/05_fairness_audit.py` and figures. Crucially, run the audit TWICE using the dataset's two label columns: once on `human_score` (fair → expect ~no disparity, the audit's null case) and once on `human_score_biased` (expect 4/5ths ≈ 0.77 → adverse impact). Show the mitigation closing the gap on the biased run, and validate the corrected scores against `true_quality`. This 'no false alarm + catches real bias + fixes it' arc is the headline result."

**Phase 6 — Explainability.**
> "Implement `src/explain/attributions.py`: token attributions + HTML phrase highlighting for a given response."

**Phase 7 — App.**
> "Build `app/streamlit_app.py` with the three tabs in Section 6.10."

**Phase 8 — Polish.**
> "Write a strong README.md (Section 11), ensure `make test` passes, add a `.gitignore`, and a one-command demo path."

---

## 9. EVALUATION METHODOLOGY (be able to defend every choice)

- **Primary metric: Quadratic Weighted Kappa (QWK).** Standard for ordinal human-vs-model agreement; penalizes large disagreements quadratically. Know its formula and why it beats plain accuracy for ordinal scores.
- **Secondary:** MAE, Pearson/Spearman correlation, within-1 accuracy.
- **Always report baseline vs transformer** — shows disciplined iteration.
- **Fairness eval is first-class, not an appendix.** Report per-group QWK/MAE, selection-rate ratios (4/5ths), and parity/equal-opportunity/equalized-odds differences.
- **Leakage guardrail:** the no-leakage test must pass; mention it unprompted in interviews.

---

## 10. STRETCH GOALS (only if time)
- LoRA/QLoRA fine-tuning toggle (already wired in config).
- LLM-as-a-judge scorer as a third model + agreement analysis with humans.
- `mlflow` experiment tracking.
- Calibration of predicted scores; confidence estimates → route low-confidence to "human review" (mirrors SHL's Augmented Intelligence).
- Dockerfile.

---

## 11. README CONTENTS (so it looks production-grade)
1. One-line pitch + a GIF/screenshot of the app.
2. Problem framing (rubric-anchored response scoring; why it's hard; why fairness matters in hiring).
3. Architecture diagram (Section 4).
4. Results table (baseline vs transformer: QWK/MAE).
5. **Fairness findings** (link `reports/fairness_report.md`) + mitigation before/after.
6. Explainability example (highlighted response).
7. How to run (unzip the dataset bundle into the project root, then `make setup && make baseline && make transformer && make eval && make fairness && make app`).
8. **Honest limitations** (synthetic sensitive attribute; template-assembled text; research prototype; no psychometric validity claim) — this honesty is itself a credibility signal.
9. Tech stack + rationale.

---

## 12. INTERVIEW TALKING POINTS (memorize)
- "I always baseline first — embeddings + GBM — before reaching for a fine-tuned transformer, so I know what the complexity is buying me."
- "I used QWK because it's the field standard for ordinal human-agreement and, unlike accuracy, it doesn't reward a model that's 'close' the same as one that's 'exact'."
- "The bias in my data lives in the labels, not the inputs — the group attribute is independent of quality — which mirrors the real finding that human raters themselves can be biased. My audit correctly finds nothing on the fair labels and catches clear adverse impact on the biased ones."
- "The fairness audit isn't an afterthought: I measured subgroup error, ran a 4/5ths adverse-impact check, and showed a post-hoc threshold mitigation closing the gap — because in hiring a biased score is a wrong, legally-exposed decision, not just a metric."
- "I added phrase-level highlighting so a human evaluator can see *why* — which is the Augmented-Intelligence, human-in-the-loop pattern rather than full automation."

---

## 13. MASTER KICKOFF PROMPT (paste this into Claude Code first)

```
You are helping me build "Automated Response Scorer with Fairness Audit," an NLP
project. I've placed the full spec in SPEC.md — read it fully before doing anything.

Context: I'm an ML engineer preparing for a Research Engineer (AI) interview at SHL,
a talent-assessment company. This project mirrors their core problem: scoring
open-ended responses against a rubric, accurately AND fairly, with explainability.

DATA IS PRE-BUILT: the synthetic dataset already exists in data/ (from
response_scorer_dataset.zip) — 4,800 rows, pre-split into data/processed/{train,val,test}.csv,
with columns including `human_score` (fair label), `human_score_biased` (biased label), and
`group`. Do NOT generate or overwrite data. Your first action in Phase 0 is to confirm those
files exist; if they don't, tell me to unzip the bundle and stop.

Build it in PHASES exactly as defined in Section 8 of SPEC.md. Do ONE phase at a time.
After each phase: summarize what you built, tell me the exact commands to run it, and
WAIT for me to confirm before starting the next phase.

Constraints:
- Python 3.11, stack per Section 3. Keep it runnable on CPU/free Colab (small models).
- Clean, typed, documented code. Pytest where specified.
- Treat the fairness module (Section 6.8) as the centerpiece — make it rigorous and
  produce a readable reports/fairness_report.md. Run it on BOTH label columns
  (human_score = fair, human_score_biased = biased) per Phase 5.
- Use only the provided synthetic data (and optionally a public set like ASAP). Never
  assume access to real candidate data.

Start with Phase 0 (scaffold). Go.
```
