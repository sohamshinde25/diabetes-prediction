"""Unit tests for single-patient and batch prediction engine."""

import pandas as pd
from src.predict import predict_risk, predict_batch, validate_patient_inputs


def test_predict_risk_single_patient():
    """Verify single patient prediction produces expected structure and ranges."""
    healthy_patient = {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 22.0,
        "DiabetesPedigreeFunction": 0.25,
        "Age": 25,
    }

    result = predict_risk(healthy_patient, model_type="random_forest")

    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_level"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert "risk_percentage" in result
    assert result["predicted_class"] in [0, 1]


def test_risk_increases_with_hyperglycemia_and_obesity():
    """Verify high glucose and BMI yield higher predicted risk than normal."""
    healthy = {
        "Pregnancies": 0,
        "Glucose": 85,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 70,
        "BMI": 21.0,
        "DiabetesPedigreeFunction": 0.15,
        "Age": 24,
    }
    high_risk = {
        "Pregnancies": 5,
        "Glucose": 190,
        "BloodPressure": 90,
        "SkinThickness": 35,
        "Insulin": 250,
        "BMI": 38.0,
        "DiabetesPedigreeFunction": 1.2,
        "Age": 55,
    }

    res_healthy = predict_risk(healthy)
    res_high = predict_risk(high_risk)

    assert res_high["probability"] > res_healthy["probability"]
    assert len(res_high["contributing_factors"]) > 0


def test_validation_warnings():
    """Verify input validation catches biologically improbable values."""
    extreme_patient = {
        "Glucose": 450,  # Extreme
        "Age": 150,      # Improbable
        "BMI": 95.0,     # Extreme
    }
    warnings = validate_patient_inputs(extreme_patient)
    assert len(warnings) >= 3


def test_predict_batch():
    """Verify batch prediction processes DataFrames and appends risk columns."""
    sample_data = pd.DataFrame({
        "Pregnancies": [0, 6],
        "Glucose": [90, 180],
        "BloodPressure": [70, 85],
        "SkinThickness": [20, 35],
        "Insulin": [80, 200],
        "BMI": [22.0, 35.0],
        "DiabetesPedigreeFunction": [0.2, 0.9],
        "Age": [25, 52],
    })

    batch_res = predict_batch(sample_data, model_type="random_forest")
    assert "Diabetes_Risk_Probability" in batch_res.columns
    assert "Risk_Tier" in batch_res.columns
    assert len(batch_res) == 2
