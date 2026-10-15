"""Unit tests for single-patient and batch prediction engine."""

import pandas as pd
import pytest
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


def test_model_prediction_on_contrasting_reference_profiles():
    """Verify the trained model assigns higher risk probability to an elevated profile vs baseline.

    NOTE: Non-linear tree ensembles do not provide strict mathematical monotonicity guarantees
    without explicit monotonicity constraints. This test validates a concrete software/model property:
    that for two contrasting synthetic benchmark profiles (normative baseline vs elevated risk factors),
    the model's response surface correctly ranks the elevated profile higher.
    """
    baseline_profile = {
        "Pregnancies": 0,
        "Glucose": 85,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 70,
        "BMI": 21.0,
        "DiabetesPedigreeFunction": 0.15,
        "Age": 24,
    }
    elevated_profile = {
        "Pregnancies": 5,
        "Glucose": 190,
        "BloodPressure": 90,
        "SkinThickness": 35,
        "Insulin": 250,
        "BMI": 38.0,
        "DiabetesPedigreeFunction": 1.2,
        "Age": 55,
    }

    res_baseline = predict_risk(baseline_profile)
    res_elevated = predict_risk(elevated_profile)

    assert res_elevated["probability"] > res_baseline["probability"]
    assert len(res_elevated["contributing_factors"]) > 0


def test_predict_risk_handles_invalid_string_values_gracefully():
    """Verify single-patient prediction coerces string inputs and handles invalid non-numerics without crashing."""
    patient_with_strings = {
        "Pregnancies": "1",
        "Glucose": "120",
        "BloodPressure": "70",
        "SkinThickness": "invalid_str",  # Non-numeric should be treated as NaN and imputed
        "Insulin": 100,
        "BMI": "28.5",
        "DiabetesPedigreeFunction": 0.45,
        "Age": 35,
    }
    res = predict_risk(patient_with_strings)
    assert 0.0 <= res["probability"] <= 1.0


def test_validation_warnings():
    """Verify input validation catches biologically improbable values and negative inputs."""
    extreme_patient = {
        "Glucose": 450,  # Extreme
        "Age": 150,      # Improbable
        "BMI": -5.0,     # Negative
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


def test_predict_batch_out_of_order_columns():
    """Verify batch prediction succeeds when columns are presented in arbitrary order."""
    permuted_data = pd.DataFrame({
        "Age": [30, 45],
        "BMI": [24.0, 31.0],
        "Glucose": [95, 140],
        "Pregnancies": [1, 3],
        "BloodPressure": [70, 80],
        "SkinThickness": [20, 30],
        "Insulin": [85, 150],
        "DiabetesPedigreeFunction": [0.3, 0.6],
    })
    batch_res = predict_batch(permuted_data, model_type="random_forest")
    assert len(batch_res) == 2
    assert "Diabetes_Risk_Probability" in batch_res.columns


def test_predict_batch_missing_columns_raises():
    """Verify ValueError is raised when required columns are absent in batch data."""
    incomplete_data = pd.DataFrame({
        "Glucose": [100, 120],
        "BMI": [25.0, 30.0],
    })
    with pytest.raises(ValueError, match="missing required columns"):
        predict_batch(incomplete_data)


def test_predict_batch_empty_dataframe_raises():
    """Verify ValueError is raised when batch DataFrame is empty."""
    with pytest.raises(ValueError, match="empty DataFrame"):
        predict_batch(pd.DataFrame())
