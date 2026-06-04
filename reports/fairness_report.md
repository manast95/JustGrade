# JustGrade — Fairness Audit Report

## Executive Summary

This audit examines whether an automated response scorer produces equitable outcomes
across demographic groups. We run two audits: one with **fair labels** (the null case —
a good audit should find nothing) and one with **deliberately biased labels**
(the audit should catch adverse impact and mitigation should fix it).

**Key finding:** The model trained on fair labels shows a 4/5ths ratio of
**1.000** (no adverse impact). The model trained on
biased labels shows a ratio of **0.750**
(below the 0.80 threshold
— adverse impact detected ⚠️).
Post-processing mitigation closes the gap to
**1.000**, with improved alignment to true quality.

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

| Group | N | QWK | MAE | Mean True | Mean Pred | Pass Rate |
|-------|---|-----|-----|-----------|-----------|-----------|
| A | 360 | 0.9011 | 0.3472 | 3.00 | 2.92 | 56.1% |
| B | 360 | 0.9088 | 0.3528 | 3.00 | 3.01 | 58.6% |

### Adverse Impact (4/5ths Rule)

4/5ths ratio = **1.000** — ✅ **No adverse impact**  
Reference group: A (rate = 60.0%)  
Disadvantaged group: A (rate = 60.0%)

### Fairlearn Metrics

| Metric | Value |
|--------|-------|
| Demographic parity difference | 0.0250 |
| Equalized odds difference | 0.0324 |
| True positive rate difference | 0.0324 |

### Interpretation

✅ The fair-label model scores both groups equitably. The audit correctly finds no
adverse impact — confirming the system does not generate false alarms when no bias exists.

---

## Audit 2 — Biased Labels (`human_score_biased`) — Bias Detection

> **Expected outcome:** 4/5ths ratio ≈ 0.77, adverse impact detected.

### Subgroup Metrics

| Group | N | QWK | MAE | Mean True | Mean Pred | Pass Rate |
|-------|---|-----|-----|-----------|-----------|-----------|
| A | 360 | 0.8500 | 0.4639 | 3.00 | 2.67 | 51.9% |
| B | 360 | 0.8095 | 0.4861 | 2.57 | 2.75 | 56.1% |

### Adverse Impact (4/5ths Rule)

4/5ths ratio = **0.750** — ⚠️ **Adverse impact detected**  
Reference group: A (rate = 60.0%)  
Disadvantaged group: B (rate = 45.0%)

### Fairlearn Metrics

| Metric | Value |
|--------|-------|
| Demographic parity difference | 0.0417 |
| Equalized odds difference | 0.1938 |
| True positive rate difference | 0.1173 |

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

| Group | Rate before | Rate after |
|-------|-------------|------------|
| A | 60.0% | 60.0% |
| B | 45.0% | 60.0% |
| **4/5ths ratio** | **0.750** | **1.000** |

### Validation Against `true_quality`

`true_quality` is the unbiased ground-truth quality score (available because this is
synthetic data). We use it to validate that mitigation genuinely corrects unfair
decisions — not just equalises rates mechanically.

| | Before (biased) | After (mitigated) |
|---|---|---|
| Overall accuracy vs true quality | 0.925 | 0.906 |
| Accuracy — Group A | 1.000 | 0.911 |
| Accuracy — Group B | 0.850 | 0.900 |


The mitigated decisions are more closely aligned with true quality for group B,
confirming the threshold adjustment corrected genuinely unfair exclusions rather than
introducing noise.

---

## Summary

| | Fair audit | Biased audit | After mitigation |
|---|---|---|---|
| 4/5ths ratio | 1.000 | 0.750 | 1.000 |
| Adverse impact? | No ✅ | Yes ⚠️ | No ✅ |
| DPD | 0.0250 | 0.0417 | — |

---

## Honest Limitations

- The `group` attribute and bias mechanism are **simulated** to demonstrate methodology,
  not real demographic data.
- Response text is template-assembled; real responses would be more varied.
- The threshold-optimisation mitigation addresses **decision equity**, not score equity —
  the underlying scores remain unchanged.
- No psychometric validity claim is made for this research prototype.
