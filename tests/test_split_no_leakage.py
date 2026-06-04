"""Verify zero ID overlap across train/val/test splits and schema validation."""
from __future__ import annotations

import pytest
from pathlib import Path

DATA_ROOT = Path("data/processed")


@pytest.mark.skipif(not (DATA_ROOT / "train.csv").exists(), reason="data not present")
def test_no_id_overlap() -> None:
    """No row ID should appear in more than one split."""
    from src.data.ingest import load_split  # noqa: PLC0415

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    train_ids = set(train["id"])
    val_ids = set(val["id"])
    test_ids = set(test["id"])

    assert train_ids.isdisjoint(val_ids), "ID overlap between train and val"
    assert train_ids.isdisjoint(test_ids), "ID overlap between train and test"
    assert val_ids.isdisjoint(test_ids), "ID overlap between val and test"


@pytest.mark.skipif(not (DATA_ROOT / "train.csv").exists(), reason="data not present")
def test_schema_validation_passes() -> None:
    """validate_schema should not raise on the shipped data."""
    from src.data.ingest import load_split  # noqa: PLC0415

    for split in ("train", "val", "test"):
        df = load_split(split)  # validate_schema is called inside load_split
        assert len(df) > 0, f"{split} split is empty"


@pytest.mark.skipif(not (DATA_ROOT / "train.csv").exists(), reason="data not present")
def test_get_label_fair_and_biased() -> None:
    """get_label should return both human_score and human_score_biased correctly."""
    from src.data.ingest import load_split, get_label  # noqa: PLC0415

    df = load_split("test")
    fair = get_label(df, "human_score")
    biased = get_label(df, "human_score_biased")

    assert fair.between(1, 5).all(), "Fair labels out of [1,5]"
    assert biased.between(1, 5).all(), "Biased labels out of [1,5]"
    # The two columns differ (bias is injected into group B)
    assert not fair.equals(biased), "Fair and biased labels are identical — check the dataset"


@pytest.mark.skipif(not (DATA_ROOT / "train.csv").exists(), reason="data not present")
def test_split_sizes() -> None:
    """Splits should total 4,800 rows in 70/15/15 proportions (approx)."""
    from src.data.ingest import load_split  # noqa: PLC0415

    n_train = len(load_split("train"))
    n_val = len(load_split("val"))
    n_test = len(load_split("test"))
    total = n_train + n_val + n_test

    assert total == 4800, f"Expected 4800 total rows, got {total}"
    assert n_train > n_val, "train should be larger than val"
    assert n_val == n_test, "val and test should be the same size"
