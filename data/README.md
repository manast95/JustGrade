# Project 1 — Synthetic Response-Scoring Dataset

A purpose-built dataset for the **Automated Response Scorer + Fairness Audit**. It contains
free-text answers to competency questions, each with a quality score and a (simulated)
group attribute, so you can train a scorer **and** run a fairness audit on the same data.

> **`group` is a SIMULATED attribute** used to demonstrate fairness methodology. It makes no
> claim about any real demographic group. All responses are synthetically generated.

## Files
```
data/
├── generate_synthetic.py        # the generator (reproducible; re-run to scale/tune)
├── raw/responses.csv            # full dataset (4,800 rows)
├── processed/train.csv          # 3,360 rows  (70%)
├── processed/val.csv            #   720 rows  (15%)
├── processed/test.csv           #   720 rows  (15%)
└── README.md                    # this file
```

## Schema (per row)
| Column | Meaning |
|---|---|
| `id` | unique row id (`r00001`…) |
| `question_id` | one of 12 competency questions (`Q01`…`Q12`) |
| `question` | the prompt text |
| `rubric` | scoring criteria for that question |
| `group` | simulated sensitive attribute, **A** or **B** (assigned at random, independent of quality) |
| `true_quality` | latent intended quality 1–5 (a synthetic-only luxury) |
| `human_score` | **FAIR** observed label (== `true_quality`). **Default training/eval label.** |
| `human_score_biased` | **BIASED** observed label (group B systematically under-scored) |
| `response` | the free-text answer |

## How it was generated (and why the signal is learnable)
Each response is assembled so its **text quality genuinely scales with the score** — this is
what lets a model learn:
- **Score 1:** very short, vague or off-topic, hedged ("I guess… nothing special really").
- **Score 2:** short, generic, minimal detail.
- **Score 3:** partial STAR (situation + action), relevant but unquantified.
- **Score 4:** mostly complete STAR, structured, some specifics.
- **Score 5:** full STAR + a concrete metric + a reflection sentence + ownership language.
Length, STAR completeness, specificity/metrics, and competency vocabulary all increase with
the score. ~35% "blur" (text drawn from an adjacent level) adds realistic label noise so
results are not artificially perfect.

**Validated signal:** a trivial TF-IDF + logistic baseline reaches **QWK ≈ 0.92** on the fair
labels — confirming the signal is strongly learnable, with headroom for a fine-tuned
transformer. (See the learnability check you can re-run in the notebook.)

## The fairness mechanism (modeled on SHL's own research)
The `group` attribute is assigned **independently of quality**, so the data is *fair by
construction*. The bias lives in the **labels**, mirroring SHL's published finding that the
human ground truth itself can be biased ("To Trust, or Not to Trust"):
- `human_score` (fair): groups A and B have **identical** score distributions →
  **4/5ths selection-rate ratio = 1.00** (no adverse impact). This is the audit's *null case*:
  a good audit should find **nothing** here.
- `human_score_biased`: group B is under-scored at mid/high quality →
  **4/5ths ratio ≈ 0.77 (< 0.80 → adverse impact)**. This is what the audit should **catch**,
  and what a mitigation (reweighing or per-group threshold adjustment) should **fix**.

You can validate any mitigation against `true_quality`, which is the unbiased ground truth.

## Recommended usage in the project
1. **Accuracy headline:** the project spec recommends a *real* dataset (e.g., ASAP) for the
   main QWK numbers, because real data is harder and more credible. This synthetic set is
   intentionally cleanly-separable.
2. **Fairness audit (this set's main job):** train two scorers — one on `human_score` (fair),
   one on `human_score_biased` — and run the audit on each. Show that it correctly finds **no
   bias** in the fair case and **clear bias** in the biased case, then apply a mitigation and
   re-measure. That "no false alarm + catches real bias + fixes it" arc is the strongest
   interview story.

## Split strategy & leakage
Split is **stratified on (question, score, group)** so every question appears in all splits —
appropriate because the task is "score a response to a *known* question against its rubric."
Each row is uniquely generated; **ID overlap across splits is 0** (verified). Note: because
text is template-assembled, some short low-quality responses can recur verbatim by chance —
this is a known property of synthetic data; on real data you would split by question/candidate.
If you prefer the stricter setup, re-run with split-by-question (the project's `split.py`
supports it).

## Regenerate / scale / tune
```bash
python data/generate_synthetic.py --per_cell 40     # 4,800 rows (default)
python data/generate_synthetic.py --per_cell 80     # 9,600 rows (more data)
```
Knobs inside the script: `maybe_blur(blur=...)` (label noise / difficulty),
`biased_label(p_down=...)` (bias strength). Everything is seeded (`seed=42`) for reproducibility.

## Honest limitations (state these in your project README too)
- Synthetic, template-assembled text — fluent but less varied than real responses.
- The sensitive attribute and the bias are simulated to demonstrate *method*, not real groups.
- Use alongside a real dataset for credible accuracy numbers; use this for the controllable,
  unambiguous fairness demonstration.
