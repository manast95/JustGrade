"""Phase 4: Full evaluation — metrics, confusion matrices, model-card table."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import load_split, get_label
from src.eval.evaluate import (
    evaluate_model,
    plot_calibration,
    plot_confusion_matrix,
    write_model_card,
)
from src.models.baseline import BaselineScorer
from src.models.transformer import TransformerScorer
from src.utils.config import get
from src.utils.logging import get_logger

logger = get_logger("evaluate")


def main() -> None:
    label_col = get("data.label_col", "human_score")
    test_df = load_split("test")
    y_true = get_label(test_df, label_col).to_numpy()

    for name, cls, path in [
        ("baseline",    BaselineScorer,    "models/baseline"),
        ("transformer", TransformerScorer, "models/transformer"),
    ]:
        model_path = Path(path)
        if not model_path.exists():
            logger.warning(f"No saved model at '{path}' — skipping {name}.")
            continue

        logger.info(f"Evaluating {name} …")
        model = cls.load(path)
        # Predict once, reuse for both metrics and the confusion matrix.
        y_pred = model.predict(test_df)
        evaluate_model(model, test_df, label_col=label_col, model_name=name,
                       y_pred=y_pred)
        plot_confusion_matrix(y_true, y_pred, model_name=name)
        plot_calibration(y_true, y_pred, model_name=name)

    print("\n" + "=" * 55)
    print("MODEL CARD")
    print("=" * 55)
    card = write_model_card()
    print(card)

    # Write the model card to a file
    card_path = Path("reports") / "model_card.md"
    card_path.write_text(
        f"# JustGrade — Model Card\n\n"
        f"**Task:** Rubric-anchored response scoring (1–5 ordinal scale)  \n"
        f"**Data:** 4,800 synthetic responses across 12 competency questions  \n"
        f"**Label:** `{label_col}` (fair labels)  \n"
        f"**Evaluation split:** 720 held-out test examples  \n\n"
        f"## Results\n\n{card}\n\n"
        f"## Notes\n\n"
        f"- **QWK** (Quadratic Weighted Kappa) is the primary metric — "
        f"field standard for ordinal scoring, penalises large disagreements quadratically.\n"
        f"- **Baseline:** `BAAI/bge-small-en-v1.5` embeddings + GradientBoosting regressor.\n"
        f"- **Transformer:** `distilbert-base-uncased` fine-tuned regression head (4 epochs, "
        f"SmoothL1 loss, early stopping on val QWK).\n"
        f"- **QWK CI** is a bootstrap 95% confidence interval (case resampling).\n"
        f"- Confusion matrix and calibration figures saved in `reports/figures/`.\n"
    )
    logger.info(f"Model card written to {card_path}")


if __name__ == "__main__":
    main()
