# Diabetes Risk Prediction

An end-to-end machine learning project exploring diabetes classification on the Pima Indians Diabetes dataset. I built this project to understand how to handle tabular healthcare data properly—specifically focusing on data quality anomalies, preventing data leakage with scikit-learn pipelines, and honestly evaluating model trade-offs between linear and tree-based approaches.

---

## Why I Built This

The Pima Indians Diabetes dataset is one of the most common beginner datasets in machine learning, but many tutorial implementations take shortcuts:
- They treat all zeros as valid numbers, ignoring that a blood pressure or glucose level of zero is biologically impossible in a living person.
- They fit scalers or imputers on the whole dataset before splitting into train and test sets, which leaks test information into training.
- They report accuracy alone without discussing recall or false negatives, which is the most dangerous metric to overlook in disease screening.

I wanted to build a clean, modular repository that handles these problems correctly:
1. Feature-specific missing value handling that respects human physiology.
2. Leakage-free preprocessing pipelines where imputation and scaling are fitted strictly on training data.
3. Decoupled preprocessing where tree-based models are not unnecessarily scaled.
4. An honest analysis of model trade-offs (including why a 53.7% recall matters).
5. An interactive Streamlit web app for single-patient and batch CSV screening.

---

## Dataset & Context

* **Source**: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), 1988 (Smith et al., 1988; hosted by UCI Machine Learning Repository).
* **Cohort**: 768 female subjects of Pima Indian heritage, aged 21 or older, living near Phoenix, Arizona.
* **Target (`Outcome`)**: Binary indicator (1 = tested positive for diabetes within 5 years of examination, 0 = tested negative).
* **Class Distribution**: 500 negative cases (65.1%) and 268 positive cases (34.9%).

### The Zero-Value Nuance
Looking closely at the summary statistics reveals that several continuous features have minimum values of zero:
- **`Pregnancies`**: 111 zeros (14.5%). A value of zero represents a woman who has never been pregnant (nulliparous). **This is biologically valid and must be kept as 0.**
- **`Glucose`**: 5 zeros (0.7%). Plasma glucose cannot be 0 in a living person.
- **`BloodPressure`**: 35 zeros (4.6%). Diastolic pressure cannot be 0 during an exam.
- **`SkinThickness`**: 227 zeros (29.6%). Skinfold thickness cannot be 0.
- **`Insulin`**: 374 zeros (48.7%). Circulating serum insulin is never zero; nearly half the cohort had this test omitted.
- **`BMI`**: 11 zeros (1.4%). Weight and height are strictly positive.

For these five features, zeros represent missing measurements. The pipeline converts them to `NaN` and imputes them using training-set medians. Median imputation was chosen because features like insulin and skinfold thickness are right-skewed with heavy tails.

### Known Limitations
* **Demographic Specificity**: All subjects are adult females of Pima Indian heritage. The models cannot be assumed to generalize to males, children, or other ethnic populations without external validation.
* **Missing Modern Biomarkers**: Modern clinical diagnostics rely heavily on Glycated Hemoglobin (HbA1c) and fasting C-peptide, which were not collected in this 1988 study.
* **Unknown Missingness Mechanism**: The original documentation does not explain why specific participants had missing insulin or skinfold measurements. We cannot assume data is missing completely at random (MCAR).
* **Correlation vs. Causation**: This is an observational study. Feature importance reflects statistical associations in this dataset, not causal clinical advice.

---

## Pipeline Architecture

```
data/diabetes.csv (768 rows)
   │
   ├── Schema Validation & Zero Cleaning (src/data_loader.py)
   │     • Pregnancies = 0 kept
   │     • Glucose, BP, SkinThickness, Insulin, BMI zeros -> NaN
   │
   ├── Stratified Train/Test Split (80/20, seed=42)
   │     • Train set: 614 rows (34.85% positive)
   │     • Test set: 154 rows (35.06% positive, isolated until final evaluation)
   │
   ├── Model-Specific Pipelines (src/preprocessing.py)
   │     • Logistic Regression: SimpleImputer(median) + StandardScaler()
   │     • Random Forest: SimpleImputer(median) only (trees are scale-invariant)
   │
   ├── 5-Fold Stratified Cross-Validation on Training Set (src/model_trainer.py)
   │     • Folds fit imputation inside each fold to avoid cross-fold leakage
   │
   └── Local Inference & Web Interface (src/predict.py, app.py)
         • Single patient scoring with range validation
         • Batch CSV cohort screening
         • Feature sensitivity exploration
```

### Why Decouple Scaling?
Logistic Regression uses gradient-based optimization with an L2 penalty, so features with larger ranges would be penalized more heavily without `StandardScaler`. 

Random Forest, on the other hand, evaluates split points based only on feature rank order ($x_j \le \theta$). Monotonic scaling has no effect on tree splits. Omitting scaling for Random Forest keeps the pipeline cleaner and avoids redundant computation.

---

## Verified Results

### 1. Held-Out Test Set Evaluation ($N=154$, 20% Stratified Split)

The held-out test set was evaluated once at the very end as an unbiased generalization check:

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

### 3. Confusion Matrix Analysis (Random Forest on Test Set)

```
                 Predicted Negative   Predicted Positive
Actual Negative          86 (TN)              14 (FP)
Actual Positive          25 (FN)              29 (TP)
```

**Key Takeaway on Recall (53.7%)**:  
At the default classification threshold of 0.50, Random Forest identifies 29 diabetic patients but misses 25. In healthcare screening, a False Negative (missing someone who has the disease) is generally far more serious than a False Positive (which leads to a confirmatory lab test).

This low recall happens because:
1. The dataset has class imbalance (~35% positive).
2. The default threshold of 0.50 favors precision over sensitivity.

If this were being deployed in a screening clinic, we would lower the decision threshold (e.g., to 0.35) to capture more true positives, accepting more false alarms in exchange for fewer missed cases.

### 4. Feature Importance
Across both models, **Plasma Glucose** (~36% RF, ~43% LR) and **BMI** (~17% RF, ~25% LR) are consistently the strongest statistical predictors, followed by Age and the Diabetes Pedigree Function. This aligns with clinical understanding that fasting blood sugar and adiposity correlate strongly with insulin resistance.

---

## Project Structure

```
diabetes-prediction/
├── data/
│   └── diabetes.csv                     # Pima Indians Diabetes Dataset
├── docs/
│   ├── DATASET_DOCUMENTATION.md         # Detailed provenance, ranges, and zero audit
│   └── INTERVIEW_PREPARATION.md         # Interview questions, architecture, and defense guide
├── models/
│   ├── logistic_regression.joblib       # Fitted Logistic Regression pipeline
│   ├── random_forest.joblib             # Fitted Random Forest pipeline
│   └── metrics.json                     # Verified evaluation metrics
├── notebooks/
│   └── diabetes_analysis.ipynb          # Executed Jupyter analysis notebook with EDA & plots
├── scripts/
│   └── build_notebook.py                # Notebook generation script
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Data loading, schema validation, zero-value handling
│   ├── preprocessing.py                 # Decoupled preprocessing pipelines
│   ├── model_trainer.py                 # Cross-validation, training, artifact saving
│   └── predict.py                       # Single-patient and batch inference engine
├── tests/
│   ├── test_data_loader.py              # Schema, bounds, and zero-preservation tests
│   ├── test_preprocessing.py            # Preprocessing steps and train-only leakage tests
│   ├── test_model.py                    # Architecture and stratification tests
│   └── test_predict.py                  # Single-patient, batch, and validation tests
├── .gitignore                           # Standard Python/macOS ignore rules
├── app.py                               # Interactive Streamlit web app
├── pytest.ini                           # Pytest configuration
├── requirements.txt                     # Curated core dependencies
└── README.md
```

---

## How to Run Locally

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/sohamshinde25/diabetes-prediction.git
cd diabetes-prediction

# Create and activate a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
pytest -v
```

All 23 unit tests should pass in under 2 seconds.

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. It includes:
* **Patient Risk Calculator**: Live risk estimation across heuristic probability bands.
* **Model Sensitivity Exploration**: Visualizes how adjusting inputs shifts the model's output probability along the learned boundary.
* **Batch CSV Screening**: Upload a cohort CSV, score all records, and download the annotated file.
* **Model Transparency Tab**: View verified metrics, confusion matrices, and feature importance.

### 4. Retrain Models

```bash
python src/model_trainer.py
```

---

## Disclaimer

This project is an educational machine learning exploration built for academic study and technical placement evaluation. It is not an FDA-approved medical diagnostic tool and does not provide clinical diagnoses, prognosis, or medical advice.
