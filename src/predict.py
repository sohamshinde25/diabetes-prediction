"""Inference and risk scoring engine for Diabetes Risk Prediction.

Provides single-patient risk evaluation, batch scoring, input validation,
and clinical risk factor contextualization.
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
    """Validate patient features against biological reference ranges.

    Returns:
        List of warning/validation messages (empty if all are within bounds).
    """
    warnings = []
    for col, (min_val, max_val) in FEATURE_RANGES.items():
        if col in inputs:
            val = inputs[col]
            if val is not None and not np.isnan(val):
                if val < min_val or val > max_val:
                    warnings.append(
                        f"Value for {col} ({val}) is outside typical biological range ({min_val} - {max_val})."
                    )
    return warnings


def predict_risk(
    patient_data: Dict[str, Any],
    model_type: str = "random_forest",
) -> Dict[str, Any]:
    """Calculate diabetes risk probability and clinical assessment for a single individual.

    Args:
        patient_data: Dictionary containing patient clinical measurements.
        model_type: Which trained classifier to use ('random_forest' or 'logistic_regression').

    Returns:
        Dictionary with probability, risk category, contributing factors, and clinical advice.
    """
    pipeline = load_model(model_type)

    # Format into DataFrame with correct columns
    row_data = {}
    for col in DEFAULT_FEATURE_NAMES:
        val = patient_data.get(col, np.nan)
        if col in ZERO_AS_MISSING_COLS and val == 0:
            val = np.nan
        row_data[col] = [val]

    df_patient = pd.DataFrame(row_data)

    proba = float(pipeline.predict_proba(df_patient)[0, 1])
    predicted_class = int(pipeline.predict(df_patient)[0])

    # Clinical risk classification
    if proba < 0.35:
        risk_level = "Low Risk"
        color = "#10b981"  # Emerald
        summary = "Model indicators suggest low probability of diabetes. Standard healthy lifestyle maintenance advised."
    elif proba < 0.65:
        risk_level = "Moderate Risk"
        color = "#f59e0b"  # Amber
        summary = "Borderline clinical markers detected. Regular screening and preventative lifestyle consultation recommended."
    else:
        risk_level = "High Risk"
        color = "#ef4444"  # Crimson
        summary = "Elevated risk profile detected. Comprehensive clinical evaluation (HbA1c / OGTT) by a healthcare professional strongly recommended."

    # Identify individual risk factors
    contributing_factors = []
    glucose = patient_data.get("Glucose", 0)
    bmi = patient_data.get("BMI", 0)
    age = patient_data.get("Age", 0)
    bp = patient_data.get("BloodPressure", 0)
    pedigree = patient_data.get("DiabetesPedigreeFunction", 0)

    if glucose and glucose >= 140:
        contributing_factors.append(f"Elevated Plasma Glucose ({glucose} mg/dL; impaired glucose tolerance or diabetic range)")
    elif glucose and glucose >= 100:
        contributing_factors.append(f"Borderline Fasting Glucose ({glucose} mg/dL)")

    if bmi and bmi >= 30.0:
        contributing_factors.append(f"Obese BMI category ({bmi:.1f} kg/m²; metabolic strain)")
    elif bmi and bmi >= 25.0:
        contributing_factors.append(f"Overweight BMI category ({bmi:.1f} kg/m²)")

    if age and age >= 45:
        contributing_factors.append(f"Age factor ({age} years; risk increases significantly over 45)")

    if bp and bp >= 90:
        contributing_factors.append(f"Elevated Diastolic Blood Pressure ({bp} mm Hg)")

    if pedigree and pedigree >= 0.8:
        contributing_factors.append(f"High Genetic Predisposition score ({pedigree:.2f})")

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
    """
    pipeline = load_model(model_type)
    df_copy = df.copy()

    # Pre-clean zero-as-missing
    for col in ZERO_AS_MISSING_COLS:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].replace(0, np.nan)

    # Ensure required feature columns exist
    features_present = [c for c in DEFAULT_FEATURE_NAMES if c in df_copy.columns]
    if len(features_present) != len(DEFAULT_FEATURE_NAMES):
        missing = set(DEFAULT_FEATURE_NAMES) - set(features_present)
        raise ValueError(f"Batch dataset missing required columns: {missing}")

    X = df_copy[DEFAULT_FEATURE_NAMES]
    probas = pipeline.predict_proba(X)[:, 1]

    result_df = df.copy()
    result_df["Diabetes_Risk_Probability"] = np.round(probas * 100, 1)
    result_df["Risk_Tier"] = pd.cut(
        probas,
        bins=[-0.01, 0.35, 0.65, 1.01],
        labels=["Low Risk", "Moderate Risk", "High Risk"],
    )

    return result_df
