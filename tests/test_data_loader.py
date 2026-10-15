"""Unit tests for data loading, schema validation, and clinical missing value handling."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    load_raw_data,
    prepare_features_and_target,
    validate_dataset_schema,
    ZERO_AS_MISSING_COLS,
    EXPECTED_COLUMNS,
)


def test_load_raw_data_success():
    """Verify raw dataset is loaded with correct shape, types, and expected columns."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 768
    assert df.shape[1] == 9
    for col in EXPECTED_COLUMNS:
        assert col in df.columns
    # Verify cohort minimum age criterion (Pima study: >= 21 years)
    assert df["Age"].min() >= 21


def test_load_raw_data_invalid_path():
    """Verify FileNotFoundError is raised for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("non_existent_data.csv")


def test_validate_dataset_schema_empty():
    """Verify ValueError is raised on empty DataFrame."""
    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataset_schema(pd.DataFrame())


def test_validate_dataset_schema_missing_columns():
    """Verify ValueError is raised when required columns are absent."""
    df = pd.DataFrame({"Pregnancies": [1, 2], "Glucose": [100, 110]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset_schema(df)


def test_validate_dataset_schema_invalid_target():
    """Verify ValueError is raised if Outcome contains values outside {0, 1}."""
    df = load_raw_data().copy()
    df.loc[0, "Outcome"] = 2  # Invalid class
    with pytest.raises(ValueError, match="Invalid target values"):
        validate_dataset_schema(df)


def test_validate_dataset_schema_negative_values():
    """Verify ValueError is raised if any physical measurement is negative."""
    df = load_raw_data().copy()
    df.loc[0, "Glucose"] = -10
    with pytest.raises(ValueError, match="Negative values detected"):
        validate_dataset_schema(df)


def test_prepare_features_and_target_zero_conversion_and_preservation():
    """Verify impossible zeros become NaN, while legitimate zeros (Pregnancies) are preserved."""
    df = load_raw_data()
    raw_zero_pregnancies = (df["Pregnancies"] == 0).sum()
    assert raw_zero_pregnancies == 111  # 111 nulliparous women in dataset

    X, y = prepare_features_and_target(df, convert_zeroes_to_nan=True)

    # 1. Target separation
    assert "Outcome" not in X.columns
    assert len(y) == 768
    assert set(y.unique()) == {0, 1}

    # 2. Legitimate zero preservation: Pregnancies = 0 must NOT be converted to NaN
    assert (X["Pregnancies"] == 0).sum() == 111
    assert X["Pregnancies"].isna().sum() == 0

    # 3. Impossible zero conversion: exactly the 5 clinical measurement columns
    expected_zero_counts = {
        "Glucose": 5,
        "BloodPressure": 35,
        "SkinThickness": 227,
        "Insulin": 374,
        "BMI": 11,
    }
    for col, orig_zeros in expected_zero_counts.items():
        assert (X[col] == 0).sum() == 0, f"Column '{col}' still has zeroes!"
        assert X[col].isna().sum() == orig_zeros, f"Column '{col}' expected {orig_zeros} NaNs, found {X[col].isna().sum()}"


def test_prepare_features_and_target_no_zero_conversion():
    """Verify zero values are preserved in all columns when conversion is disabled."""
    df = load_raw_data()
    X, y = prepare_features_and_target(df, convert_zeroes_to_nan=False)
    assert (X["Insulin"] == 0).sum() == 374
    assert (X["Glucose"] == 0).sum() == 5
