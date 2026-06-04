# JustGrade — Model Card

**Task:** Rubric-anchored response scoring (1–5 ordinal scale)  
**Data:** 4,800 synthetic responses across 12 competency questions  
**Label:** `human_score` (fair labels)  
**Evaluation split:** 720 held-out test examples  

## Results

| Metric | Baseline | Transformer |
|--------|-------:|-------:|
| QWK (↑) | 0.9050 | **0.9308** |
| MAE (↓) | 0.3500 | **0.2764** |
| Pearson r (↑) | 0.9053 | **0.9308** |
| Spearman ρ (↑) | 0.9054 | **0.9308** |
| Within-1 acc (↑) | 0.9847 | **1.0000** |
| QWK 95% CI | [0.8907, 0.9183] | [0.9206, 0.9405] |

## Notes

- **QWK** (Quadratic Weighted Kappa) is the primary metric — field standard for ordinal scoring, penalises large disagreements quadratically.
- **Baseline:** `BAAI/bge-small-en-v1.5` embeddings + GradientBoosting regressor.
- **Transformer:** `distilbert-base-uncased` fine-tuned regression head (4 epochs, SmoothL1 loss, early stopping on val QWK).
- **QWK CI** is a bootstrap 95% confidence interval (case resampling).
- Confusion matrix and calibration figures saved in `reports/figures/`.
