"""Data loading and validation module for Diabetes Risk Prediction.

This module loads the publicly available Pima Indians Diabetes dataset,
validates schema consistency, and replaces biologically impossible zero values
with NaN for appropriate imputation during preprocessing.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

# Columns where 0 represents missing/unrecorded physiological values
ZERO_AS_MISSING_COLS: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

EXPECTED_COLUMNS: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# Physiologically plausible clinical reference ranges for validation
FEATURE_RANGES = {
    "Pregnancies": (0, 25),
    "Glucose": (40, 300),  # mg/dL
    "BloodPressure": (30, 200),  # mm Hg
    "SkinThickness": (5, 100),  # mm
    "Insulin": (5, 900),  # mu U/ml
    "BMI": (10.0, 75.0),  # kg/m^2
    "DiabetesPedigreeFunction": (0.05, 3.0),
    "Age": (15, 120),  # years
}


def load_raw_data(data_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw diabetes CSV dataset from disk.

    Args:
        data_path: Path to the diabetes CSV file. Defaults to 'data/diabetes.csv'.

    Returns:
        pd.DataFrame containing the raw records.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing.
    """
    if data_path is None:
        data_path = Path(__file__).resolve().parent.parent / "data" / "diabetes.csv"
    else:
        data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    df = pd.read_csv(data_path)

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    convert_zeroes_to_nan: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target variable and optionally replace impossible zeros with NaN.

    In the Pima Indians Diabetes dataset, clinical variables like Glucose,
    BloodPressure, SkinThickness, Insulin, and BMI cannot biologically be zero
    in a living patient. These values indicate missing data.

    Args:
        df: Input DataFrame containing features and Outcome column.
        convert_zeroes_to_nan: Whether to convert impossible zeros to NaN.

    Returns:
        Tuple of (X features DataFrame, y target Series).
    """
    df_copy = df.copy()

    if convert_zeroes_to_nan:
        for col in ZERO_AS_MISSING_COLS:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].replace(0, np.nan)

    X = df_copy.drop(columns=["Outcome"])
    y = df_copy["Outcome"].astype(int)

    return X, y
