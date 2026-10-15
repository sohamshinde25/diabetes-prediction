"""Preprocessing pipeline definitions for Diabetes Risk Prediction.

Constructs reusable scikit-learn transformers and pipelines to ensure zero
data leakage between train and test sets.

Methodology Decisions:
1. Median Imputation:
   - Used for missing physiological features (Glucose, Blood Pressure,
     Skin Thickness, Insulin, BMI).
   - Median is chosen over mean due to the right-skewed distribution of
     Insulin and Skin Thickness.
2. Standard Scaling:
   - REQUIRED for Logistic Regression: L2 regularization and gradient descent
     optimizers require features to have comparable variance (zero mean, unit variance).
   - OMITTED for Random Forest: Decision trees split features based on monotonic
     rank ordering (x_j <= threshold) and are strictly invariant to monotonic feature
     scaling. Applying StandardScaler to Random Forest adds unnecessary computation
     with zero mathematical effect on tree split decisions.
"""

from typing import List
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURE_NAMES: List[str] = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def create_preprocessing_pipeline(include_scaler: bool = True) -> Pipeline:
    """Create a scikit-learn preprocessing pipeline.

    Args:
        include_scaler: If True, appends StandardScaler after SimpleImputer.
                        Set to True for linear/distance models (Logistic Regression).
                        Set to False for tree-based models (Random Forest).

    Returns:
        Configured sklearn.pipeline.Pipeline instance.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if include_scaler:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps=steps)


def create_logistic_regression_preprocessor() -> Pipeline:
    """Preprocessor for Logistic Regression: Median Imputer + StandardScaler."""
    return create_preprocessing_pipeline(include_scaler=True)


def create_random_forest_preprocessor() -> Pipeline:
    """Preprocessor for Random Forest: Median Imputer only (scale-invariant)."""
    return create_preprocessing_pipeline(include_scaler=False)
