"""Load and validate the pre-built dataset splits."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import get, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {
    "id", "question_id", "question", "rubric", "group",
    "true_quality", "human_score", "human_score_biased", "response",
}

LABEL_COLUMNS = {"true_quality", "human_score", "human_score_biased"}


def validate_schema(df: pd.DataFrame) -> None:
    """Assert required columns exist, labels in configured range, no nulls.

    Raises:
        ValueError: on any schema violation.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    lo, hi = get("label_range", [1, 5])
    for col in LABEL_COLUMNS:
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if not out_of_range.empty:
            raise ValueError(
                f"Column '{col}' has {len(out_of_range)} values outside [{lo}, {hi}]"
            )

    null_counts = df[list(REQUIRED_COLUMNS)].isnull().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"Null values found: {bad.to_dict()}")


def load_split(name: str) -> pd.DataFrame:
    """Load data/processed/{train,val,test}.csv and validate schema.

    Args:
        name: one of 'train', 'val', 'test'

    Returns:
        Validated DataFrame.

    Raises:
        FileNotFoundError: if the CSV is missing.
        ValueError: if the schema is invalid.
    """
    cfg = load_config()
    key = f"{name}_path"
    path_str = cfg.get("data", {}).get(key)
    if path_str is None:
        # Fallback: construct conventional path
        path_str = f"data/processed/{name}.csv"

    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Split '{name}' not found at '{path}'. "
            "Unzip response_scorer_dataset.zip into the project root."
        )

    df = pd.read_csv(path)
    validate_schema(df)

    lo, hi = get("label_range", [1, 5])
    logger.info(
        f"Loaded {name:>5}: {len(df):,} rows  "
        f"human_score {lo}–{hi} distribution: "
        + "  ".join(f"{k}→{v}" for k, v in
                    df["human_score"].value_counts().sort_index().items())
    )
    return df


def get_label(df: pd.DataFrame, label_col: str | None = None) -> pd.Series:
    """Return the chosen label column as a Series.

    Args:
        df: a loaded split DataFrame.
        label_col: column name. Defaults to config.data.label_col ('human_score').

    Returns:
        pd.Series of integer labels.

    Raises:
        ValueError: if label_col is not in the DataFrame.
    """
    if label_col is None:
        label_col = get("data.label_col", "human_score")

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available: {list(df.columns)}"
        )
    return df[label_col]
