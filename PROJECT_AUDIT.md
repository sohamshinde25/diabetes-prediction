# Comprehensive Project & Code Audit: Diabetes Risk Prediction

**Project**: Diabetes Risk Prediction  
**Auditor**: Senior Machine Learning Engineer & Technical Interview Reviewer  
**Repository**: [https://github.com/sohamshinde25/diabetes-prediction](https://github.com/sohamshinde25/diabetes-prediction)  
**Date**: October 2026  
**Status**: Pre-Improvement Audit (Phase 1)

---

## Executive Summary

This repository contains an end-to-end machine learning project classifying diabetes outcomes using the historical Pima Indians Diabetes dataset. The repository structure is clean, modular, and possesses working pipelines and tests.

However, from the standpoint of **technical interviews (AIML / Software Engineering placements)** and **rigorous ML engineering standards**, several critical issues must be resolved before this repository can be confidently defended:

1. **Unsupported Medical Claims & Causal Overreach**: The project uses terms like "Clinical-Grade" and "Production-Ready", and the Streamlit app implements a "What-If" slider that claims causal risk reduction (e.g., "-15 mg/dL Glucose reduces risk by X%"). On observational data, correlation is not causation. An interviewer will heavily penalize this.
2. **Methodological Inconsistencies**: Feature scaling (`StandardScaler`) is applied to Random Forest where it is mathematically redundant; and Random Forest was selected as the "winner" based on the held-out test set even though Logistic Regression scored higher in Cross-Validation.
3. **Unaddressed False Negatives (Recall ~53.7%)**: Test recall is only 53.7% (25 false negatives out of 54 actual diabetic patients). In healthcare screening, missing ~46% of positive cases is a critical vulnerability that must be analyzed technically.
4. **Dependency & Environment Bloat**: `requirements.txt` is an uncurated 128-line `pip freeze` dump containing platform-specific packages (`appnope`, `ptyprocess`).
5. **Branding & Promotional Language**: Over-promoted "$0.00 / 100% Free" badges throughout the UI and README distract from the engineering substance.

Below is the complete itemized audit detailing findings, severity, code evidence, and proposed resolutions.

---

## Detailed Audit Findings

### 1. Inappropriate "Clinical-Grade" Claims and Causal "What-If" Simulation
- **Severity**: **CRITICAL** (Major Technical Interview Red Flag)
- **Code Evidence**:
  - `README.md` (lines 9, 15, 196): *"An end-to-end clinical machine learning application...", "Clinical Data Engineering", "Dual Clinical ML Pipeline"*.
  - `app.py` (lines 199–202): *"🩺 GlucoSense Risk AI: Clinical-grade Machine Learning for early Type 2 Diabetes screening and prevention analytics"*.
  - `app.py` (lines 384–388, 455–462): The "What-If" intervention module computes:
    ```python
    diff = prob_pct - sim_prediction["risk_percentage"]
    st.info(f"With a {wi_glucose_drop} mg/dL glucose reduction... the patient's risk drops from {prob_pct}% to {sim_prediction['risk_percentage']}% (-{diff:.1f}% absolute risk reduction).")
    ```
- **Technical Analysis**:
  - Observational models predict conditional probabilities $P(Y=1 \mid X)$, NOT interventional probabilities $P(Y=1 \mid do(X))$. Perturbing inputs does not estimate the causal effect of medical intervention.
  - The dataset consists exclusively of 768 adult female Pima Indians from a 1988 epidemiological study. It cannot be characterized as "clinical grade" for the general population.
- **Proposed Fix**:
  - Reframe the project strictly as an educational/exploratory ML classification project on historical benchmark data.
  - Replace "Clinical-Grade" and "Production-Ready" with objective technical descriptions.
  - Reframe the "What-If" slider explicitly as a **"Model Sensitivity & Feature Response Surface"** exploration tool, accompanied by an explicit note that associative models do not imply causal clinical treatment effects.

---

### 2. Model Selection Discrepancy (Cross-Validation vs. Test Set Leakage)
- **Severity**: **HIGH** (ML Methodology Trap)
- **Code Evidence**:
  - `models/metrics.json` & `notebooks/diabetes_analysis.ipynb`:
    - **Logistic Regression CV ROC-AUC**: `0.843 ± 0.019`, **CV Accuracy**: `78.8% ± 1.9%`
    - **Random Forest CV ROC-AUC**: `0.835 ± 0.021`, **CV Accuracy**: `77.7% ± 1.8%`
    - **Held-Out Test Set Accuracy**: RF = `74.7%`, LR = `70.8%`
  - In `README.md` and `notebooks/diabetes_analysis.ipynb`, Random Forest was declared the "Primary Recommendation" based on test set accuracy (74.7%).
- **Technical Analysis**:
  - Deciding the "superior" model based on test set score violates strict ML methodology (test set snooping / contamination). The test set must be reserved solely for final evaluation.
  - On the validation splits (5-fold CV), Logistic Regression actually outperformed Random Forest across both ROC-AUC and Accuracy. Random Forest's test set advantage is within normal sample variation ($N_{\text{test}}=154$).
- **Proposed Fix**:
  - Document both models objectively. Highlight that Logistic Regression is the stronger and more parsimonious baseline on cross-validation, while Random Forest demonstrates how non-linear tree ensembles perform.
  - Explain the difference between CV model selection and final test verification in the documentation and interview prep.

---

### 3. Redundant Feature Scaling on Tree-Based Model
- **Severity**: **MEDIUM** (Technical Precision)
- **Code Evidence**:
  - `src/preprocessing.py` (lines 24–43):
    ```python
    def create_preprocessing_pipeline() -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    ```
  - `src/model_trainer.py` (lines 51, 72): Both `logistic_regression` and `random_forest` use this exact same preprocessing pipeline.
- **Technical Analysis**:
  - Decision trees and Random Forests evaluate orthogonal splits $X_j \le \theta$ based on rank order and are completely invariant to monotonic transformations like `StandardScaler`.
  - While mathematically harmless to tree accuracy, applying `StandardScaler` to Random Forest without explanation suggests the author may not understand decision tree mechanics.
- **Proposed Fix**:
  - Decouple the pipelines or parameterize `create_preprocessing_pipeline(include_scaler=True/False)`. Apply scaling to Logistic Regression only, and imputer-only to Random Forest.
  - Document this choice clearly as a talking point for interviews.

---

### 4. Low Recall (~53.7%) and Unaddressed False Negatives
- **Severity**: **HIGH** (Clinical Domain & ML Evaluation)
- **Code Evidence**:
  - `models/metrics.json` (lines 91, 99):
    ```json
    "recall": 0.5370370370370371,
    "confusion_matrix": {"tn": 86, "fp": 14, "fn": 25, "tp": 29}
    ```
- **Technical Analysis**:
  - At the default classification threshold ($p \ge 0.50$), Random Forest misses 25 out of 54 diabetic individuals (a False Negative Rate of 46.3%).
  - In medical screening, False Negatives carry high clinical risk (undiagnosed condition leading to progression).
  - The project currently does not explain why recall is low (class imbalance: 34.9% positive class prevalence, combined with default 0.5 cutoff) or discuss threshold calibration vs. `class_weight="balanced"`.
- **Proposed Fix**:
  - Add a dedicated technical discussion in `docs/` and `notebooks/` detailing the Precision-Recall trade-off.
  - Analyze how threshold tuning shifts Sensitivity vs. Specificity without claiming that changing the threshold makes the model "clinically safe".

---

### 5. Input Validation, Range Checking & Error Handling Gaps
- **Severity**: **MEDIUM** (Software Reliability)
- **Code Evidence**:
  - `src/predict.py` (lines 46–54): `validate_patient_inputs` produces warning strings but does not prevent downstream errors if non-numeric inputs or extreme values are passed.
  - `src/predict.py` (lines 74–78): If a key is missing from `patient_data`, it silently defaults to `np.nan` and is imputed without warning.
  - In `predict_batch` (lines 161–165): Missing columns raise a generic `ValueError`, but extra columns or non-numeric types are not gracefully sanitized.
- **Technical Analysis**:
  - Production prediction services must validate input schemas, reject negative ages/blood pressures, cast numeric types defensively, and report actionable errors.
- **Proposed Fix**:
  - Implement robust input validation in `src/predict.py` with type coercion, boundary checks, and informative exceptions.

---

### 6. Dependency Bloat in `requirements.txt`
- **Severity**: **MEDIUM** (Portability & Repository Hygiene)
- **Code Evidence**:
  - `requirements.txt` contains 128 lines generated from an uncurated `pip freeze`.
  - Includes macOS-only packages like `appnope==1.0.0` and low-level build packages like `ptyprocess`, `terminado`, and `fastjsonschema`.
- **Technical Analysis**:
  - Installing these on Linux/Windows or constrained environments can cause wheel compilation failures or dependency conflicts.
- **Proposed Fix**:
  - Clean `requirements.txt` to list direct runtime dependencies with sensible version pins (`streamlit`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `matplotlib`, `seaborn`).
  - Provide a clear dev section or separate requirements file for testing (`pytest`) and notebook editing (`jupyter`).

---

### 7. Overly Promotional "$0 Budget" Badges & Marketing Language
- **Severity**: **LOW-MEDIUM** (Professionalism & Aesthetics)
- **Code Evidence**:
  - `README.md` (lines 6, 9, 146–163): Budget badge, repeated claims of "$0.00 / 100% Free".
  - `app.py` (lines 205, 588–631): Dedicated "Cost & Architecture Audit" tab and hero badge.
- **Technical Analysis**:
  - Portfolio reviewers and hiring managers look for engineering rigor, clean code, and solid statistics. Emphasizing "$0 cost" repeatedly distracts from the machine learning work.
- **Proposed Fix**:
  - Remove promotional badges.
  - State once, neutrally, in the README that the project runs locally and relies exclusively on open-source dependencies.

---

### 8. Test Suite Coverage Gaps
- **Severity**: **MEDIUM** (Test Quality)
- **Code Evidence**:
  - `tests/` currently contains 11 tests that test only basic functionality.
  - Missing tests:
    - Out-of-order column handling in batch prediction.
    - Handling non-numeric and negative values.
    - Missing value edge cases (all features missing/NaN).
    - Reproducibility checks (verifying identical predictions on fixed input across runs).
- **Proposed Fix**:
  - Expand the test suite in `tests/` to 15+ comprehensive unit tests covering edge cases, input sanitization, and pipeline invariance.

---

## Proposed Remediation Plan by Phases

| Phase | Focus Area | Proposed Changes |
|---|---|---|
| **Phase 2** | **Dataset & Data Quality** | Verify dataset provenance (UCI/NIDDK 1988); document feature-specific zero-handling decisions; document population limitations (Pima Indian women $\ge 21$); update data loader documentation. |
| **Phase 3** | **ML Pipeline & Leakage Audit** | Decouple preprocessing so `StandardScaler` is only applied to Logistic Regression; verify strict train/test featurization order; confirm CV folds avoid cross-fold leakage. |
| **Phase 4** | **Model Evaluation** | Re-run and verify metrics in `models/metrics.json`; document CV vs test set results; provide in-depth analysis of 53.7% recall, confusion matrix, false negatives, and threshold trade-offs; clarify feature importance is associative, not causal. |
| **Phase 5** | **Prediction Engine & App Polish** | Harden `src/predict.py` with robust input validation; rebrand "What-If" simulation to "Model Sensitivity Exploration" with clear non-causal disclaimer; remove exaggerated "clinical-grade" and budget marketing from `app.py`. |
| **Phase 6** | **Testing Suite** | Expand `pytest` suite to cover edge cases, type errors, out-of-order batch columns, and pipeline reproducibility. |
| **Phase 7** | **README Rewrite** | Produce a concise, technical, recruiter-friendly `README.md` following the requested structure without promotional badges or unverified medical claims. |
| **Phase 8** | **Interview Documentation** | Create `docs/INTERVIEW_PREPARATION.md` covering the 30s/90s/3m elevator pitches, architecture diagrams, 25+ detailed technical Q&As, and design defense. |
| **Phase 9** | **Final Validation** | End-to-end verification of all tests, model retraining, and local Streamlit execution. |
| **Phase 10**| **GitHub Push** | Clean Git commit and push to existing `origin/main` without history disruption. |

---

## Action Required

Please review the findings and proposed remediation plan above. Once approved, we will proceed systematically starting with **Phase 2 (Dataset and Data Quality)**.
