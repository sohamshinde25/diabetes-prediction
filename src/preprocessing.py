"""Preprocessing pipeline definitions for Diabetes Risk Prediction.

Constructs reusable scikit-learn transformers and pipelines to ensure zero
data leakage between train and test sets.
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


def create_preprocessing_pipeline() -> Pipeline:
    """Create a standard scikit-learn preprocessing pipeline.

    The pipeline performs:
    1. Median imputation for missing values (represented as NaN).
       Median is chosen due to the skewed clinical distributions of Insulin,
       SkinThickness, and Glucose.
    2. Standard scaling (zero mean, unit variance) so that features have
       equal numerical footing for regularized models like Logistic Regression.

    Returns:
        Configured sklearn.pipeline.Pipeline instance.
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline
