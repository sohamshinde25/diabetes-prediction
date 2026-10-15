"""Unit tests for data loading and feature preparation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    load_raw_data,
    prepare_features_and_target,
    ZERO_AS_MISSING_COLS,
    EXPECTED_COLUMNS,
)


def test_load_raw_data_success():
    """Verify raw dataset is loaded with correct shape and columns."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 768
    for col in EXPECTED_COLUMNS:
        assert col in df.columns


def test_load_raw_data_invalid_path():
    """Verify FileNotFoundError is raised for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("non_existent_data.csv")


def test_prepare_features_and_target_zero_conversion():
    """Verify zero values in physiological columns are converted to NaN."""
    df = load_raw_data()
    X, y = prepare_features_and_target(df, convert_zeroes_to_nan=True)

    assert "Outcome" not in X.columns
    assert len(y) == 768
    assert set(y.unique()) == {0, 1}

    # Verify that invalid zeros are replaced by NaN in features
    for col in ZERO_AS_MISSING_COLS:
        assert (X[col] == 0).sum() == 0, f"Column {col} still has zeroes!"
        assert X[col].isna().sum() > 0, f"Column {col} should have NaN after conversion"


def test_prepare_features_and_target_no_zero_conversion():
    """Verify zero values are preserved if convert_zeroes_to_nan is False."""
    df = load_raw_data()
    X, y = prepare_features_and_target(df, convert_zeroes_to_nan=False)
    assert (X["Insulin"] == 0).sum() > 0
