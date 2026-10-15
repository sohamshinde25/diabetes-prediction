"""Data loading, schema validation, and clinical missing value handling.

Dataset: Pima Indians Diabetes Database
Origin: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), 1988.
Cohort: Females of Pima Indian heritage, aged >= 21, residing near Phoenix, Arizona.

This module handles:
1. Strict schema and column existence verification.
2. Biological sanity checks (e.g. non-negative values, binary target).
3. Feature-specific missing value handling:
   - Valid zeroes (e.g. Pregnancies = 0 for nulliparous women) are explicitly preserved.
   - Biologically impossible zeroes (Glucose, Blood Pressure, Skin Thickness,
     Insulin, BMI) are converted to NaN for downstream imputation without data leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Columns where 0 represents missing/unrecorded physiological measurements
# rather than legitimate biological zeros.
# NOTE: 'Pregnancies' is deliberately NOT in this list, because 0 pregnancies
# is a completely valid biological state for nulliparous women.
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

# Physiologically plausible biological reference ranges for validation
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "Pregnancies": (0, 25),
    "Glucose": (40, 300),  # mg/dL
    "BloodPressure": (30, 200),  # mm Hg
    "SkinThickness": (5, 100),  # mm
    "Insulin": (5, 900),  # mu U/ml
    "BMI": (10.0, 75.0),  # kg/m^2
    "DiabetesPedigreeFunction": (0.05, 3.0),
    "Age": (21, 120),  # years (cohort entry criterion: >= 21 years)
}


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """Validate raw DataFrame schema and fundamental biological bounds.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If required columns are missing, data is empty,
                    or values violate fundamental integrity constraints.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Verify target variable is strictly binary {0, 1}
    target_values = set(df["Outcome"].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"Invalid target values in 'Outcome': {target_values}. Expected subset of {{0, 1}}.")

    # Check for impossible negative numbers in all non-negative physical features
    numeric_cols = [c for c in EXPECTED_COLUMNS if c != "Outcome"]
    for col in numeric_cols:
        if (df[col] < 0).any():
            raise ValueError(f"Negative values detected in strictly non-negative column: '{col}'.")


def load_raw_data(data_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw diabetes CSV dataset from disk and validate schema.

    Args:
        data_path: Path to the diabetes CSV file. Defaults to 'data/diabetes.csv'.

    Returns:
        pd.DataFrame containing the raw records.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing or validation fails.
    """
    if data_path is None:
        data_path = Path(__file__).resolve().parent.parent / "data" / "diabetes.csv"
    else:
        data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    df = pd.read_csv(data_path)
    validate_dataset_schema(df)
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    convert_zeroes_to_nan: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate target variable and optionally replace impossible zeros with NaN.

    Feature-Specific Zero Handling:
    - Glucose, BloodPressure, SkinThickness, Insulin, BMI: Zero is biologically
      impossible in a living patient; zeroes are replaced by NaN.
    - Pregnancies: Zero is completely valid (nulliparous women) and is preserved.
    - Outcome: Binary target (0 = negative, 1 = positive) is preserved.

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
