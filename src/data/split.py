"""Optional re-split utility (data ships pre-split — only implement if asked)."""
from __future__ import annotations

import pandas as pd


def make_splits(
    df: pd.DataFrame,
    by: str = "stratified",
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val, test).

    by='stratified' — stratify on question×score×group (default, shipped split method).
    by='question'   — strict question-disjoint splits (harder generalization test).

    Returns:
        (train_df, val_df, test_df)
    """
    raise NotImplementedError("Only implement if re-splitting is requested")
