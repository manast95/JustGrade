"""Phase 2: Train the embedding + regressor baseline and log metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import load_split, get_label
from src.eval.metrics import compute_all
from src.models.baseline import BaselineScorer
from src.utils.config import get
from src.utils.logging import get_logger

logger = get_logger("train_baseline")
REPORTS = Path("reports")
MODELS = Path("models/baseline")


def main() -> None:
    label_col = get("data.label_col", "human_score")
    lo, hi = get("label_range", [1, 5])

    logger.info("Loading data …")
    train_df = load_split("train")
    test_df  = load_split("test")

    logger.info("Training baseline …")
    model = BaselineScorer()
    model.fit(train_df, label_col=label_col)

    logger.info("Evaluating on test split …")
    y_pred = model.predict(test_df)
    y_true = get_label(test_df, label_col).to_numpy()
    metrics = compute_all(y_true, y_pred, min_rating=lo, max_rating=hi)

    print("\n── Baseline results ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")

    # Merge into existing metrics.json (don't overwrite other model results)
    REPORTS.mkdir(exist_ok=True)
    metrics_path = REPORTS / "metrics.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing["baseline"] = {"label_col": label_col, **metrics}
    metrics_path.write_text(json.dumps(existing, indent=2))
    logger.info(f"Metrics written to {metrics_path}")

    model.save(str(MODELS))
    logger.info(f"Model saved to {MODELS}/")


if __name__ == "__main__":
    main()
