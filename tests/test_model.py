"""Unit tests for model training and evaluation."""

import json
from pathlib import Path
from src.model_trainer import build_model_pipeline, train_and_save_all_models


def test_build_model_pipelines():
    """Verify that both model architectures build properly."""
    lr_pipe = build_model_pipeline("logistic_regression")
    assert "preprocessor" in lr_pipe.named_steps
    assert "classifier" in lr_pipe.named_steps

    rf_pipe = build_model_pipeline("random_forest")
    assert "preprocessor" in rf_pipe.named_steps
    assert "classifier" in rf_pipe.named_steps


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
