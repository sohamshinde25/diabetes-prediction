"""Unit tests for preprocessing pipeline."""

import numpy as np
import pandas as pd
from src.preprocessing import create_preprocessing_pipeline


def test_preprocessing_pipeline_imputation_and_scaling():
    """Verify preprocessing imputes missing values and standardizes features."""
    pipeline = create_preprocessing_pipeline()

    # Create dummy data with NaNs
    dummy_data = pd.DataFrame({
        "Pregnancies": [1.0, 3.0, 2.0, 5.0],
        "Glucose": [120.0, np.nan, 140.0, 90.0],
        "BloodPressure": [70.0, 80.0, np.nan, 72.0],
        "SkinThickness": [20.0, 25.0, 30.0, np.nan],
        "Insulin": [np.nan, 100.0, 150.0, 80.0],
        "BMI": [25.0, 31.0, np.nan, 28.0],
        "DiabetesPedigreeFunction": [0.4, 0.6, 0.2, 0.8],
        "Age": [25.0, 45.0, 35.0, 50.0],
    })

    transformed = pipeline.fit_transform(dummy_data)

    # Output should have same shape, no NaNs
    assert transformed.shape == dummy_data.shape
    assert not np.isnan(transformed).any()

    # Standard scaling should produce approximate 0 mean for columns
    means = np.mean(transformed, axis=0)
    for m in means:
        assert abs(m) < 1e-6
