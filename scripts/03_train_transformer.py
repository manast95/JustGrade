"""Phase 3: Fine-tune DistilBERT regression head and log metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import load_split, get_label
from src.eval.metrics import compute_all
from src.models.transformer import TransformerScorer
from src.utils.config import get
from src.utils.logging import get_logger

logger = get_logger("train_transformer")
REPORTS = Path("reports")
MODELS = Path("models/transformer")


def main() -> None:
    label_col = get("data.label_col", "human_score")
    lo, hi = get("label_range", [1, 5])

    logger.info("Loading data …")
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    logger.info("Fine-tuning transformer …")
    model = TransformerScorer()
    model.fit(train_df, val_df, label_col=label_col, output_dir=str(MODELS))

    logger.info("Evaluating on test split …")
    y_pred = model.predict(test_df)
    y_true = get_label(test_df, label_col).to_numpy()
    metrics = compute_all(y_true, y_pred, min_rating=lo, max_rating=hi)

    print("\n── Transformer results ───────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<20} {v:.4f}")

    # Merge into reports/metrics.json alongside baseline
    REPORTS.mkdir(exist_ok=True)
    metrics_path = REPORTS / "metrics.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    existing["transformer"] = {"label_col": label_col, **metrics}

    # Side-by-side comparison
    if "baseline" in existing:
        print("\n── Comparison ────────────────────────────────────")
        print(f"  {'metric':<20} {'baseline':>10} {'transformer':>12}")
        print(f"  {'-'*44}")
        for k in ["qwk", "mae", "pearson", "spearman", "within_one_acc"]:
            b = existing["baseline"].get(k, float("nan"))
            t = metrics.get(k, float("nan"))
            marker = " ▲" if (k != "mae" and t > b) or (k == "mae" and t < b) else ""
            print(f"  {k:<20} {b:>10.4f} {t:>12.4f}{marker}")

    metrics_path.write_text(json.dumps(existing, indent=2))
    logger.info(f"Metrics written to {metrics_path}")

    model.save(str(MODELS))
    logger.info(f"Model saved to {MODELS}/")


if __name__ == "__main__":
    main()
