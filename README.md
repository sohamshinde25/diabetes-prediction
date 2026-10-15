# Diabetes Risk Prediction

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![Framework: Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](app.py)

An end-to-end machine learning project exploring diabetes outcome classification using the historical Pima Indians Diabetes dataset. It covers data-quality checks, leakage-aware preprocessing, model comparison, evaluation, and an interactive Streamlit demonstration.

---

## 📌 Problem Overview

Type 2 diabetes is a chronic metabolic condition characterized by insulin resistance and relative deficiency. Early identification of individuals with elevated risk patterns can assist preventative screening programs. This project investigates whether non-invasive physiological and demographic attributes can reliably predict diabetes onset within five years of examination.

### Dataset Provenance & Cohort
* **Source**: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), 1988 (Smith et al., 1988; UCI Machine Learning Repository).
* **Cohort**: 768 female subjects aged $\ge 21$ years of Pima Indian heritage living near Phoenix, Arizona.
* **Target (`Outcome`)**: Binary diagnosis ($1 = \text{positive}, 0 = \text{negative}$).
* **Prevalence**: 500 negative cases (65.10%), 268 positive cases (34.90%).

### Critical Dataset Limitations
1. **Demographic Specificity**: The study population consists exclusively of adult female Pima Indians. Results cannot be generalized across broader multi-ethnic, male, or pediatric cohorts without external recalibration.
2. **Missing Modern Biomarkers**: Foundational modern diagnostic biomarkers like Glycated Hemoglobin (HbA1c) and fasting C-peptide were not recorded in this 1988 dataset.
3. **Associative, Not Causal**: The models estimate statistical correlation on observational data; predictions do not represent causal treatment effects.

---

## ⚙️ Technical Approach & Pipeline Architecture

```
Raw CSV (768 rows)
   │
   ├── [1] Data Quality & Schema Validation (src/data_loader.py)
   │     • Preserve valid zeros: Pregnancies = 0 (111 nulliparous records)
   │     • Map impossible zeros to NaN: Glucose (5), BP (35), SkinThickness (227), Insulin (374), BMI (11)
   │
   ├── [2] Featurization Split (80/20 Stratified, seed=42)
   │     • Train set (N=614, 34.85% positives)
   │     • Held-out test set (N=154, 35.06% positives, strictly isolated)
   │
   ├── [3] Leakage-Free Preprocessing Pipelines (src/preprocessing.py)
   │     • Logistic Regression: SimpleImputer(median) + StandardScaler()
   │     • Random Forest: SimpleImputer(median) only (scale-invariant)
   │
   ├── [4] Model Training & Cross-Validation (src/model_trainer.py)
   │     • 5-Fold Stratified Cross-Validation on training set
   │     • Model persistence via Joblib (models/)
   │
   └── [5] Evaluation & Inference (src/predict.py & app.py)
         • Held-out test set benchmark
         • Streamlit web interface with sensitivity exploration & batch CSV scoring
```

### Data Leakage Safeguards
- **Strict Featurization Order**: Train/test splitting occurs before any imputer or scaler is fit.
- **Cross-Validation Isolation**: Imputation and scaling transformers are fit strictly on fold-training partitions during 5-fold cross-validation.
- **Tailored Preprocessing**: Feature standardization (`StandardScaler`) is applied to Logistic Regression (where regularization requires uniform scale) and omitted for Random Forest (which is invariant to monotonic transformations).

---

## 📊 Model Evaluation Results

### 1. Held-Out Test Set Metrics ($N=154$, 20% Stratified Split)

The held-out test set was evaluated once as an unbiased generalization check:

| Metric | Random Forest | Logistic Regression (Baseline) |
|---|---|---|
| **ROC-AUC** | **0.812** | **0.813** |
| **PR-AUC (Average Precision)** | **0.689** | 0.673 |
| **Accuracy** | **74.7%** | 70.8% |
| **Precision** | **67.4%** | 60.0% |
| **Recall (Sensitivity)** | **53.7%** | 50.0% |
| **Specificity (True Negative Rate)** | **86.0%** | 82.0% |
| **F1-Score** | **0.598** | 0.545 |

### 2. 5-Fold Cross-Validation on Training Set ($N=614$)

| Metric | Random Forest (Mean ± Std) | Logistic Regression (Mean ± Std) |
|---|---|---|
| **CV ROC-AUC** | 0.835 ± 0.020 | **0.843 ± 0.019** |
| **CV Accuracy** | 77.7% ± 1.8% | **78.8% ± 1.9%** |
| **CV Precision** | 73.3% ± 6.4% | **76.4% ± 6.0%** |
| **CV Recall** | **58.0% ± 3.0%** | 57.5% ± 2.3% |
| **CV F1-Score** | 64.5% ± 1.4% | **65.5% ± 2.4%** |

### 3. Confusion Matrix Analysis (Random Forest Test Set)

```
                 Predicted Negative   Predicted Positive
Actual Negative          86 (TN)              14 (FP)
Actual Positive          25 (FN)              29 (TP)
```

* **Understanding Recall (53.7%) and False Negatives**: At the default decision threshold ($p \ge 0.50$), Random Forest correctly identifies 29 diabetic patients but misses 25. In disease screening, false negatives represent individuals whose condition remains undetected. This moderate sensitivity is driven by class imbalance (34.9% positive rate) and threshold placement. Lowering the decision threshold increases recall (sensitivity) at the expense of false positives (lower precision).

### 4. Feature Importance (Statistical Association)
* **Leading Associations**: Plasma Glucose (~36.2% RF, ~42.7% LR) and Body Mass Index (~17.1% RF, ~24.9% LR) consistently exhibit the highest statistical association with diabetes outcome across both models.
* *Note*: Feature importance reflects empirical predictive utility within this dataset, not causal clinical evidence.

---

## 🛠️ Project Structure

```
diabetes-prediction/
├── data/
│   └── diabetes.csv                     # Historical Pima Indians Diabetes Dataset
├── docs/
│   └── DATASET_DOCUMENTATION.md         # Detailed provenance, feature ranges, and zero audit
├── models/
│   ├── logistic_regression.joblib       # Serialized Logistic Regression pipeline
│   ├── random_forest.joblib             # Serialized Random Forest pipeline
│   └── metrics.json                     # Verified cross-validation and test evaluation metrics
├── notebooks/
│   └── diabetes_analysis.ipynb          # Executed Jupyter analysis notebook with EDA & plots
├── scripts/
│   └── build_notebook.py                # Notebook generator script
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Schema validation, raw ingestion, zero handling
│   ├── preprocessing.py                 # Tailored preprocessing pipelines
│   ├── model_trainer.py                 # Cross-validation, model training, artifact persistence
│   └── predict.py                       # Single-patient and batch inference engine
├── tests/
│   ├── test_data_loader.py              # Schema, bounds, and zero-preservation tests
│   ├── test_preprocessing.py            # Pipeline steps and train-only leakage isolation tests
│   ├── test_model.py                    # Pipeline architecture and stratification tests
│   └── test_predict.py                  # Single-patient, batch, and validation tests
├── .gitignore                           # Git ignore rules
├── app.py                               # Interactive Streamlit application
├── pytest.ini                           # Pytest configuration
├── requirements.txt                     # Curated open-source dependencies
└── README.md                            # Project documentation
```

---

## 🚀 Quickstart & Reproduction

The project runs locally and relies exclusively on free, open-source libraries.

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/sohamshinde25/diabetes-prediction.git
cd diabetes-prediction

# Create and activate Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
pytest -v
```

### 3. Launch Streamlit Application

```bash
streamlit run app.py
```

The interactive application will open at `http://localhost:8501`.

### 4. Reproduce Model Training & Metrics

```bash
python src/model_trainer.py
```

---

## ⚠️ Educational & Research Disclaimer

This project is an educational machine learning exploration developed for academic and technical placement evaluation. It is not an FDA-approved medical diagnostic tool and does not provide clinical diagnoses, prognosis, or medical advice.
