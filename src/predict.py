"""Inference and risk scoring engine for Diabetes Risk Prediction.

Provides single-patient risk evaluation, batch scoring, input validation,
and statistical factor contextualization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import joblib

from src.data_loader import ZERO_AS_MISSING_COLS, FEATURE_RANGES
from src.preprocessing import DEFAULT_FEATURE_NAMES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_CACHE: Dict[str, Any] = {}


def load_model(model_type: str = "random_forest"):
    """Load a trained model pipeline from disk with in-memory caching.

    Args:
        model_type: 'logistic_regression' or 'random_forest'.

    Returns:
        Loaded sklearn Pipeline.
    """
    if model_type not in _MODEL_CACHE:
        model_path = MODELS_DIR / f"{model_type}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Please run model_trainer.py first."
            )
        _MODEL_CACHE[model_type] = joblib.load(model_path)
    return _MODEL_CACHE[model_type]


def validate_patient_inputs(inputs: Dict[str, Any]) -> List[str]:
    """Validate patient features against biological reference ranges and numeric constraints.

    Returns:
        List of warning/validation messages (empty if all are within bounds).
    """
    warnings = []
    for col, (min_val, max_val) in FEATURE_RANGES.items():
        if col in inputs:
            raw_val = inputs[col]
            if raw_val is None:
                continue
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                warnings.append(f"Value for {col} ('{raw_val}') is not a valid number.")
                continue

            if np.isnan(val):
                continue

            if val < 0:
                warnings.append(f"Value for {col} ({val}) cannot be negative.")
            elif val < min_val or val > max_val:
                warnings.append(
                    f"Value for {col} ({val}) is outside typical biological reference range ({min_val} - {max_val})."
                )
    return warnings


def predict_risk(
    patient_data: Dict[str, Any],
    model_type: str = "random_forest",
) -> Dict[str, Any]:
    """Calculate diabetes risk probability and assessment for a single individual.

    Args:
        patient_data: Dictionary containing patient clinical measurements.
        model_type: Which trained classifier to use ('random_forest' or 'logistic_regression').

    Returns:
        Dictionary with probability, risk category, contributing factors, and model summary.
    """
    pipeline = load_model(model_type)

    # Format into DataFrame with correct columns and types
    row_data = {}
    for col in DEFAULT_FEATURE_NAMES:
        raw_val = patient_data.get(col, np.nan)
        try:
            val = float(raw_val) if raw_val is not None else np.nan
        except (ValueError, TypeError):
            val = np.nan

        # Convert physiologically impossible zero measurements to NaN for pipeline imputer
        if col in ZERO_AS_MISSING_COLS and val == 0:
            val = np.nan
        row_data[col] = [val]

    df_patient = pd.DataFrame(row_data)

    proba = float(pipeline.predict_proba(df_patient)[0, 1])
    predicted_class = int(pipeline.predict(df_patient)[0])

    # Statistical risk categorization (Heuristic probability bands)
    if proba < 0.35:
        risk_level = "Low Risk"
        color = "#10b981"  # Emerald
        summary = "Estimated probability of diabetes is low based on the benchmark model."
    elif proba < 0.65:
        risk_level = "Moderate Risk"
        color = "#f59e0b"  # Amber
        summary = "Estimated probability of diabetes is in the intermediate band."
    else:
        risk_level = "High Risk"
        color = "#ef4444"  # Crimson
        summary = "Estimated probability of diabetes is in the elevated band. Note: This exploratory tool does not constitute medical diagnosis."

    # Identify notable feature contributions based on normative reference thresholds
    contributing_factors = []
    try:
        glucose = float(patient_data.get("Glucose", 0) or 0)
        bmi = float(patient_data.get("BMI", 0) or 0)
        age = float(patient_data.get("Age", 0) or 0)
        bp = float(patient_data.get("BloodPressure", 0) or 0)
        pedigree = float(patient_data.get("DiabetesPedigreeFunction", 0) or 0)

        if glucose >= 140:
            contributing_factors.append(f"Elevated Plasma Glucose ({glucose:.0f} mg/dL; post-challenge threshold)")
        elif glucose >= 100:
            contributing_factors.append(f"Borderline Plasma Glucose ({glucose:.0f} mg/dL)")

        if bmi >= 30.0:
            contributing_factors.append(f"BMI in Obese category ({bmi:.1f} kg/m²)")
        elif bmi >= 25.0:
            contributing_factors.append(f"BMI in Overweight category ({bmi:.1f} kg/m²)")

        if age >= 45:
            contributing_factors.append(f"Age demographic ({age:.0f} years)")

        if bp >= 90:
            contributing_factors.append(f"Elevated Diastolic Blood Pressure ({bp:.0f} mm Hg)")

        if pedigree >= 0.8:
            contributing_factors.append(f"High Genetic Pedigree score ({pedigree:.2f})")
    except (ValueError, TypeError):
        pass

    return {
        "probability": proba,
        "risk_percentage": round(proba * 100, 1),
        "predicted_class": predicted_class,
        "risk_level": risk_level,
        "color": color,
        "summary": summary,
        "contributing_factors": contributing_factors,
        "validation_warnings": validate_patient_inputs(patient_data),
        "model_used": model_type,
    }


def predict_batch(
    df: pd.DataFrame,
    model_type: str = "random_forest",
) -> pd.DataFrame:
    """Perform batch risk prediction on a DataFrame.

    Args:
        df: Input DataFrame containing expected feature columns.
        model_type: 'random_forest' or 'logistic_regression'.

    Returns:
        DataFrame copy with added 'Diabetes_Risk_Probability' and 'Risk_Tier' columns.

    Raises:
        ValueError: If required columns are missing or if DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Cannot perform batch prediction on an empty DataFrame.")

    # Ensure required feature columns exist
    missing_cols = [c for c in DEFAULT_FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Batch dataset missing required columns: {missing_cols}")

    pipeline = load_model(model_type)
    df_copy = df[DEFAULT_FEATURE_NAMES].copy()

    # Coerce to numeric values safely
    for col in DEFAULT_FEATURE_NAMES:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

    # Pre-clean zero-as-missing in physiological measures
    for col in ZERO_AS_MISSING_COLS:
        df_copy[col] = df_copy[col].replace(0, np.nan)

    probas = pipeline.predict_proba(df_copy)[:, 1]

    result_df = df.copy()
    result_df["Diabetes_Risk_Probability"] = np.round(probas * 100, 1)
    result_df["Risk_Tier"] = pd.cut(
        probas,
        bins=[-0.01, 0.35, 0.65, 1.01],
        labels=["Low Risk", "Moderate Risk", "High Risk"],
    )

    return result_df
