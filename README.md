# 🩺 GlucoSense — Diabetes Risk Prediction

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](tests/)
[![Budget: $0 USD](https://img.shields.io/badge/Cost-%240.00%20(100%25%20Free)-success.svg)](#-0-budget-verification)
[![Streamlit App](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)](app.py)

An end-to-end clinical machine learning application for early **Type 2 Diabetes screening and risk stratification**, built entirely with a **strict total budget of $0 USD** using open-source tools and local compute.

---

## 🌟 Key Features

1. **Dual Clinical ML Pipeline**:
   - **Random Forest Classifier** (Ensemble of 200 trees, non-linear feature interactions, ROC-AUC: **0.813**, Test Accuracy: **74.7%**, Test Precision: **67.4%**).
   - **Logistic Regression** (Interpretable, L2-regularized linear baseline, ROC-AUC: **0.813**).
2. **Clinical Data Engineering**:
   - Detection and treatment of biologically impossible zeros in physiological measures (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI).
   - Preprocessing with **Median Imputation** and **Standard Scaling** embedded into leak-free scikit-learn pipelines.
   - Strict featurization ordering (train/test split before fitting transformers).
3. **Interactive Streamlit Web Dashboard (`app.py`)**:
   - **Patient Risk Calculator**: Live risk gauge with dynamic risk tiers (*Low Risk*, *Moderate Risk*, *High Risk*).
   - **Clinical Presets**: One-click loading of *Healthy*, *Borderline*, and *Elevated Risk* profiles.
   - **What-If Simulation**: Interactively model how targeted lifestyle changes (e.g. -15 mg/dL Glucose, -3.0 kg/m² BMI) reduce absolute risk.
   - **Model Transparency**: Side-by-side metric comparison, confusion matrices, and feature importance charts.
   - **Batch CSV Cohort Screening**: Upload a cohort file, score all patients, and export the annotated CSV.
4. **Comprehensive Analysis Notebook**:
   - `notebooks/diabetes_analysis.ipynb` with complete EDA, clinical distributions, correlation matrix, cross-validation, and narrative conclusions.
5. **Automated Pytest Test Suite**:
   - 11 unit tests covering data loading, schema integrity, pipeline transformations, prediction bounds, and input validation.

---

## 📁 Repository Structure

```
diabetes-prediction/
├── data/
│   └── diabetes.csv                     # Public Pima Indians Diabetes Dataset
├── models/
│   ├── logistic_regression.joblib       # Serialized Logistic Regression pipeline
│   ├── random_forest.joblib             # Serialized Random Forest pipeline
│   └── metrics.json                     # Cross-validation and test set evaluation metrics
├── notebooks/
│   └── diabetes_analysis.ipynb          # Executed Jupyter analysis notebook with EDA & charts
├── scripts/
│   └── build_notebook.py                # Notebook generator script
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Data ingestion, schema check, missing value handling
│   ├── preprocessing.py                 # Scikit-learn preprocessing pipeline (Imputer + Scaler)
│   ├── model_trainer.py                 # Cross-validation, training, and artifact persistence
│   └── predict.py                       # Single-patient & batch scoring engine
├── tests/
│   ├── test_data_loader.py              # Data loading tests
│   ├── test_preprocessing.py            # Preprocessing pipeline tests
│   ├── test_model.py                    # Model structure & AUC threshold tests
│   └── test_predict.py                  # Single-patient and batch inference tests
├── .gitignore                           # Git ignore rules
├── app.py                               # Modern interactive Streamlit web dashboard
├── pytest.ini                           # Pytest configuration
├── requirements.txt                     # Pinned open-source dependencies
└── README.md                            # Documentation and quickstart guide
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.12 installed on your local machine.
- Git.

### 2. Setup Virtual Environment

```bash
# Clone or navigate to the project directory
cd /Users/sohamshinde/Documents/diabetes-prediction

# Create isolated Python virtual environment
python3.12 -m venv .venv

# Activate environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

### 3. Run Automated Tests

Verify pipeline stability and inference accuracy:

```bash
pytest -v
```

Expected output: `11 passed in ~1.2s`.

### 4. Launch the Interactive Web Application

Start the local Streamlit server:

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

### 5. Explore the Jupyter Notebook

To inspect the exploratory data analysis and model comparison:

```bash
jupyter notebook notebooks/diabetes_analysis.ipynb
```

---

## 📊 Model Evaluation Summary

Evaluated on a held-out test set ($N=154$, 20% stratified split):

| Metric | Random Forest (Production) | Logistic Regression (Baseline) |
|---|---|---|
| **ROC-AUC** | **0.813** | **0.813** |
| **PR-AUC (Average Precision)** | **0.689** | 0.673 |
| **Accuracy** | **74.7%** | 70.8% |
| **Precision** | **67.4%** | 60.0% |
| **Recall (Sensitivity)** | **53.7%** | 50.0% |
| **Specificity (True Negative Rate)** | **86.0%** | 82.0% |
| **F1-Score** | **0.598** | 0.545 |

**Top 3 Clinical Drivers**:
1. **Plasma Glucose** (~36% - 43% relative importance)
2. **Body Mass Index (BMI)** (~17% - 25% relative importance)
3. **Age & Diabetes Pedigree Function** (~11% - 14% relative importance)

---

## 🔒 $0 Budget Verification

This project was built from scratch strictly adhering to a **$0 USD total cost**:

| Resource Category | Tool / Service Used | Cost |
|---|---|---|
| **Runtime & Compute** | Local CPU (macOS / Python 3.12) | **$0.00** |
| **Machine Learning** | scikit-learn (free, open source) | **$0.00** |
| **Data Processing** | pandas, numpy (free, open source) | **$0.00** |
| **Visualizations** | matplotlib, seaborn (free, open source) | **$0.00** |
| **Web Dashboard** | Streamlit (free, open source, local server) | **$0.00** |
| **Model Persistence** | Joblib (local file storage) | **$0.00** |
| **Testing** | Pytest (free, open source) | **$0.00** |
| **Notebooks** | Jupyter (free, open source) | **$0.00** |
| **External AI APIs** | **None** (No OpenAI, Gemini, Claude, or paid models) | **$0.00** |
| **Cloud Services** | **None** (No AWS, GCP, Azure, or paid databases) | **$0.00** |
| **Dataset** | Pima Indians Diabetes Dataset (Public Domain / NIDDK) | **$0.00** |
| **Total Expenditure** | | **$0.00 USD** |

---

## 🌐 Publishing to GitHub (100% Free Steps)

To publish this repository to your GitHub account without paying for any service:

1. **Create a Free Repository on GitHub**:
   - Log in to [GitHub](https://github.com).
   - Click the **+** icon in the upper right corner and select **New repository**.
   - Enter `diabetes-prediction` as the Repository name.
   - Choose **Public** (always 100% free with unlimited repositories).
   - Leave "Initialize with a README" **unchecked** (we already have a local repo).
   - Click **Create repository**.

2. **Link and Push Your Local Commits**:
   Open your terminal in this directory and run:

   ```bash
   # Add your GitHub repository as remote origin
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/diabetes-prediction.git

   # Push to main branch
   git branch -M main
   git push -u origin main
   ```

3. **Free Deployment (Optional)**:
   If you ever wish to deploy the app online for free:
   - Use **Streamlit Community Cloud** (share.streamlit.io): 100% free for public GitHub repositories with zero billing or credit card requirements.

---

## ⚕️ Clinical Disclaimer

This software is developed solely for educational, research, and screening decision-support purposes. It is not an FDA-approved medical device and does not constitute medical diagnosis, treatment, or prognostic advice. Users should always consult a licensed healthcare professional for medical evaluation and diagnostic lab testing.
