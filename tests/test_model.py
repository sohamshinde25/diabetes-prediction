"""Unit tests for model training pipeline architectures, featurization ordering, and leakage safeguards."""

import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data_loader import load_raw_data, prepare_features_and_target
from src.preprocessing import DEFAULT_FEATURE_NAMES
from src.model_trainer import build_model_pipeline, train_and_save_all_models


def test_build_model_pipelines_architecture():
    """Verify that both model architectures build properly with tailored preprocessing."""
    lr_pipe = build_model_pipeline("logistic_regression")
    assert "preprocessor" in lr_pipe.named_steps
    assert "classifier" in lr_pipe.named_steps
    lr_prep_steps = [name for name, _ in lr_pipe.named_steps["preprocessor"].steps]
    assert "imputer" in lr_prep_steps
    assert "scaler" in lr_prep_steps  # Logistic Regression requires scaling

    rf_pipe = build_model_pipeline("random_forest")
    assert "preprocessor" in rf_pipe.named_steps
    assert "classifier" in rf_pipe.named_steps
    rf_prep_steps = [name for name, _ in rf_pipe.named_steps["preprocessor"].steps]
    assert "imputer" in rf_prep_steps
    assert "scaler" not in rf_prep_steps  # Random Forest is scale-invariant


def test_feature_names_order_consistency():
    """Verify feature ordering in loaded data matches DEFAULT_FEATURE_NAMES strictly."""
    df = load_raw_data()
    X, _ = prepare_features_and_target(df)
    assert list(X.columns) == DEFAULT_FEATURE_NAMES


def test_train_test_split_stratification_preserves_class_ratio():
    """Verify stratified split preserves target prevalence (~34.9% positives) in both partitions."""
    df = load_raw_data()
    X, y = prepare_features_and_target(df)
    full_pos_ratio = y.mean()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    train_pos_ratio = y_train.mean()
    test_pos_ratio = y_test.mean()

    # Must match within 1 percentage point
    assert abs(train_pos_ratio - full_pos_ratio) < 0.01
    assert abs(test_pos_ratio - full_pos_ratio) < 0.01
    assert len(X_train) == 614
    assert len(X_test) == 154


def test_models_exist_and_metrics_threshold():
    """Verify saved models exist and meet ROC-AUC >= 0.75 threshold."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    metrics_path = models_dir / "metrics.json"

    if not metrics_path.exists():
        train_and_save_all_models()

    assert (models_dir / "logistic_regression.joblib").exists()
    assert (models_dir / "random_forest.joblib").exists()
    assert metrics_path.exists()

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    for m_type in ["logistic_regression", "random_forest"]:
        assert m_type in metrics
        test_auc = metrics[m_type]["test_metrics"]["roc_auc"]
        # Models must achieve a reliable ROC-AUC >= 0.75
        assert test_auc >= 0.75, f"{m_type} AUC {test_auc} below required 0.75"
