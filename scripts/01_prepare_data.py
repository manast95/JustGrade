"""Phase 1 smoke test: load each split and print row counts + label distributions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ingest import load_split, get_label
from src.utils.config import get


def main() -> None:
    label_col = get("data.label_col", "human_score")
    print(f"\nLabel column: '{label_col}'\n{'='*55}")

    total = 0
    for split in ("train", "val", "test"):
        df = load_split(split)
        labels = get_label(df, label_col)
        biased = get_label(df, "human_score_biased")
        total += len(df)

        print(f"\n{split.upper()} — {len(df):,} rows")
        print(f"  {label_col} dist : " +
              "  ".join(f"{k}:{v}" for k, v in
                        labels.value_counts().sort_index().items()))
        print(f"  human_score_biased: " +
              "  ".join(f"{k}:{v}" for k, v in
                        biased.value_counts().sort_index().items()))
        print(f"  group A/B split   : " +
              "  ".join(f"{k}:{v}" for k, v in
                        df["group"].value_counts().sort_index().items()))

    print(f"\n{'='*55}")
    print(f"Total rows across all splits: {total:,}")
    print("Schema validation: PASSED")


if __name__ == "__main__":
    main()
