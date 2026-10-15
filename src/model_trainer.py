"""Model training and evaluation module for Diabetes Risk Prediction.

Implements strict featurization ordering, stratified cross-validation,
model training for Logistic Regression and Random Forest, comprehensive
metrics computation, and local persistence with Joblib.

Data Leakage Safeguards:
1. Featurization Ordering: train_test_split is performed BEFORE any imputer or
   scaler is fit.
2. Cross-Validation Isolation: Cross-validation runs inside scikit-learn's
   cross_validate() over the training set only. Fold imputers/scalers are fit
   strictly on fold training partitions.
3. Model-Specific Preprocessing:
   - Logistic Regression uses Median Imputation + StandardScaler.
   - Random Forest uses Median Imputation without scaling (trees are scale-invariant).
4. Held-Out Test Integrity: The test set is reserved exclusively for final
   unbiased performance evaluation, never for model selection or tuning.
"""

from pathlib import Path
import sys

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.data_loader import load_raw_data, prepare_features_and_target
from src.preprocessing import create_preprocessing_pipeline, DEFAULT_FEATURE_NAMES


def build_model_pipeline(classifier_type: str = "random_forest", random_state: int = 42) -> Pipeline:
    """Build a complete preprocessing + classifier pipeline.

    Applies tailored preprocessing:
    - 'logistic_regression': Median Imputation + Standard Scaling
    - 'random_forest': Median Imputation only (no scaling needed for tree-based models)

    Args:
        classifier_type: 'logistic_regression' or 'random_forest'.
        random_state: Seed for reproducibility.

    Returns:
        Configured Pipeline.
    """
    if classifier_type == "logistic_regression":
        preprocessor = create_preprocessing_pipeline(include_scaler=True)
        classifier = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
    elif classifier_type == "random_forest":
        preprocessor = create_preprocessing_pipeline(include_scaler=False)
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Train pipeline, perform stratified k-fold cross-validation on train set, and evaluate on held-out test set.

    Args:
        pipeline: The sklearn Pipeline to evaluate.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features (held-out, never used for training or fold selection).
        y_test: Test labels.
        cv_folds: Number of stratified cross-validation folds.

    Returns:
        Dictionary containing CV scores, test set metrics, and confusion matrix.
    """
    # Cross-validation on train split only to measure generalization without test leakage
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set once for unbiased benchmark
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": specificity,
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    cv_summary = {
        metric: {
            "mean": float(np.mean(cv_results[f"test_{metric}"])),
            "std": float(np.std(cv_results[f"test_{metric}"])),
        }
        for metric in scoring
    }

    return {
        "cv_metrics": cv_summary,
        "test_metrics": test_metrics,
    }


def extract_feature_importance(pipeline: Pipeline, feature_names: list) -> Dict[str, float]:
    """Extract feature importance (RF) or normalized absolute coefficients (LR) from a fitted pipeline."""
    clf = pipeline.named_steps["classifier"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        coefs = np.abs(clf.coef_[0])
        total = np.sum(coefs)
        importances = coefs / total if total > 0 else coefs
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    return {name: float(round(imp, 4)) for name, imp in zip(feature_names, importances)}


def train_and_save_all_models(
    data_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train both Logistic Regression and Random Forest models and persist artifacts locally.

    Returns:
        Dictionary summarizing metrics for all trained models.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "models"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(data_path)
    X, y = prepare_features_and_target(df, convert_zeroes_to_nan=True)

    # Verify feature names match DEFAULT_FEATURE_NAMES
    assert list(X.columns) == DEFAULT_FEATURE_NAMES, "Feature order mismatch with DEFAULT_FEATURE_NAMES"

    # Strict featurization ordering: split before fitting preprocessors
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    models_info = {}
    feature_names = list(X.columns)

    for m_type in ["logistic_regression", "random_forest"]:
        pipe = build_model_pipeline(classifier_type=m_type, random_state=random_state)
        eval_results = evaluate_pipeline(pipe, X_train, y_train, X_test, y_test)
        importances = extract_feature_importance(pipe, feature_names)

        # Save model pipeline
        model_file = output_dir / f"{m_type}.joblib"
        joblib.dump(pipe, model_file)

        models_info[m_type] = {
            "model_path": str(model_file.name),
            "feature_importance": importances,
            **eval_results,
        }

    # Save metadata & metrics JSON
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(models_info, f, indent=2)

    return models_info


if __name__ == "__main__":
    print("Training models...")
    results = train_and_save_all_models()
    print("Training complete! Results summary:")
    for m_type, res in results.items():
        print(f"\n--- {m_type.upper()} ---")
        print(f"Test Accuracy: {res['test_metrics']['accuracy']:.4f}")
        print(f"Test ROC-AUC:  {res['test_metrics']['roc_auc']:.4f}")
        print(f"Test Recall:   {res['test_metrics']['recall']:.4f}")
        print(f"Test F1:       {res['test_metrics']['f1']:.4f}")
