"""Unit tests for preprocessing pipeline and data leakage prevention."""

import numpy as np
import pandas as pd
from src.preprocessing import (
    create_preprocessing_pipeline,
    create_logistic_regression_preprocessor,
    create_random_forest_preprocessor,
)


def test_logistic_regression_preprocessor_has_imputer_and_scaler():
    """Verify Logistic Regression preprocessor includes both imputer and scaler."""
    pipeline = create_logistic_regression_preprocessor()
    step_names = [name for name, _ in pipeline.steps]
    assert "imputer" in step_names
    assert "scaler" in step_names

    dummy_data = pd.DataFrame({
        "Glucose": [100.0, 120.0, 140.0],
        "BMI": [20.0, 25.0, 30.0],
    })
    transformed = pipeline.fit_transform(dummy_data)
    # Output should be standardized (zero mean)
    means = np.mean(transformed, axis=0)
    for m in means:
        assert abs(m) < 1e-6


def test_random_forest_preprocessor_has_imputer_without_scaler():
    """Verify Random Forest preprocessor omits scaler because trees are scale-invariant."""
    pipeline = create_random_forest_preprocessor()
    step_names = [name for name, _ in pipeline.steps]
    assert "imputer" in step_names
    assert "scaler" not in step_names

    dummy_data = pd.DataFrame({
        "Glucose": [100.0, np.nan, 140.0],
        "BMI": [20.0, 30.0, np.nan],
    })
    transformed = pipeline.fit_transform(dummy_data)
    # Imputed Glucose should be the median (120.0), retaining its physical scale
    assert not np.isnan(transformed).any()
    assert transformed[1, 0] == 120.0  # Median imputed directly without z-score scaling


def test_preprocessing_fit_on_train_only_no_leakage():
    """Verify preprocessor fits statistics strictly on training data with zero test leakage."""
    train_df = pd.DataFrame({
        "Glucose": [100.0, 120.0, 140.0],  # Median = 120.0
    })
    test_df = pd.DataFrame({
        "Glucose": [np.nan, 500.0, 600.0],  # If leaked, median would be influenced by 500/600
    })

    imputer = create_random_forest_preprocessor()
    imputer.fit(train_df)

    # Transform test set using strictly train statistics
    transformed_test = imputer.transform(test_df)

    # The missing value in test_df must be replaced by train median (120.0), NOT test median
    assert transformed_test[0, 0] == 120.0
