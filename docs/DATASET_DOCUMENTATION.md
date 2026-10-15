# Dataset Documentation & Provenance: Pima Indians Diabetes

## 1. Dataset Overview & Provenance

| Property | Detail |
|---|---|
| **Dataset Name** | Pima Indians Diabetes Database |
| **Original Donor** | National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) |
| **Original Publication** | Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus*. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261–265). IEEE Computer Society Press. |
| **Public Repository** | UCI Machine Learning Repository / OpenML (Dataset ID 37) |
| **License / Terms of Use** | Public Domain / Open Research Benchmark Dataset |
| **Total Observations ($N$)** | 768 rows |
| **Total Predictor Features** | 8 numerical physiological attributes |
| **Target Variable** | `Outcome` (Binary: 0 = Negative, 1 = Positive) |
| **Class Distribution** | 500 Negative (65.10%), 268 Positive (34.90%) |
| **Base Rate Imbalance** | ~1.87 : 1 |

---

## 2. Cohort Demographics & Critical Limitations

### Cohort Specification
- All patients in this historical dataset are **females of Pima Indian heritage**, aged **21 years or older**, who resided near Phoenix, Arizona, and underwent diagnostic oral glucose testing following World Health Organization (WHO) criteria.

### Population & Generalizability Constraints
1. **Demographic & Ethnic Specificity**: The Pima Indian population historically experiences high prevalence of Type 2 Diabetes. Models trained on this dataset **cannot be assumed to generalize** to male patients, pediatric/adolescent cohorts, or multi-ethnic populations without external validation and recalibration.
2. **Missing Modern Clinical Biomarkers**: The study was conducted prior to the widespread clinical adoption of **Hemoglobin A1c (HbA1c)** and **fasting C-peptide**, which are foundational to modern diabetes screening and diagnostic guidelines.
3. **Observational & Cross-Sectional Nature**: The data contains observational measurements. Predictive associations learned by algorithms do **not** reflect causal treatment mechanisms.
4. **Missingness Mechanism is Unknown**: While it is known that clinical tests such as 2-hour serum insulin (48.7% zeros) and skinfold thickness (29.6% zeros) were unrecorded for many participants, the specific protocol or clinical reasons for omitted tests were not documented in the original Smith et al. (1988) study. The missingness mechanism cannot be definitively classified as MCAR (Missing Completely at Random), MAR (Missing at Random), or MNAR (Missing Not at Random) without speculative assumptions.
5. **Sample Size**: With 768 rows, the dataset is relatively small, making it susceptible to variance across train/test splits.

---

## 3. Feature Definitions & Biological Reference Ranges

| Feature Name | Data Type | Physical / Clinical Unit | Minimum in Data | Maximum in Data | Median in Data | Clinical Description |
|---|---|---|---|---|---|---|
| `Pregnancies` | Integer | Count | 0 | 17 | 3.0 | Total number of times pregnant (nulliparous women have 0). |
| `Glucose` | Integer | $\text{mg/dL}$ | 0* (min non-zero: 44) | 199 | 117.0 | Plasma glucose concentration at 2 hours in an oral glucose tolerance test (OGTT). |
| `BloodPressure` | Integer | $\text{mm Hg}$ | 0* (min non-zero: 24) | 122 | 72.0 | Diastolic blood pressure. |
| `SkinThickness` | Integer | $\text{mm}$ | 0* (min non-zero: 7) | 99 | 23.0 | Triceps skinfold thickness (caliper measurement of subcutaneous body fat). |
| `Insulin` | Integer | $\mu\text{U/mL}$ | 0* (min non-zero: 14) | 846 | 30.5 | 2-Hour serum insulin concentration. |
| `BMI` | Float | $\text{kg/m}^2$ | 0.0* (min non-zero: 18.2) | 67.1 | 32.0 | Body Mass Index ($\text{weight in kg} / (\text{height in m})^2$). |
| `DiabetesPedigreeFunction` | Float | Unitless score | 0.078 | 2.42 | 0.3725 | Scoring function synthesizing genetic risk based on family history of diabetes. |
| `Age` | Integer | Years | 21 | 81 | 29.0 | Age of patient at the time of examination (cohort entry age $\ge 21$). |
| `Outcome` | Integer | Categorical $\{0, 1\}$ | 0 | 1 | 0.0 | Diabetes diagnosis within 5 years of examination ($1=\text{positive}, 0=\text{negative}$). |

*\* Note: Indicates that 0 represents a missing measurement rather than a true biological value.*

---

## 4. Feature-by-Feature Zero-Value Audit & Rationale

A frequent mistake in naive analyses is to replace all zeros in the dataset with `NaN`, or conversely to treat all zeros as valid numbers. In this repository, we apply a strict, feature-specific rationale grounded in human physiology:

### A. Valid Zero Values (PRESERVED as 0)
* **`Pregnancies`** (111 zeros, 14.45% of dataset):
  - *Physiological Rationale*: A value of 0 is entirely valid and represents a nulliparous female (a woman who has never been pregnant). Replacing this with `NaN` would corrupt an informative, legitimate demographic feature.
  - *Decision*: **Retain 0 as valid numerical data.**
* **`Outcome`** (500 zeros, 65.10% of dataset):
  - *Physiological Rationale*: Target class 0 represents patients who tested negative for diabetes.
  - *Decision*: **Retain as class label 0.**

### B. Biologically Impossible Zeroes (CONVERTED to NaN for Imputation)
* **`Glucose`** (5 zeros, 0.65% of dataset):
  - *Physiological Rationale*: Human blood glucose cannot reach 0 mg/dL in a living ambulatory patient; severe hypoglycemia causes unconsciousness and death well before 0. A zero indicates an unrecorded test.
  - *Decision*: **Convert 0 to `NaN`.**
* **`BloodPressure`** (35 zeros, 4.56% of dataset):
  - *Physiological Rationale*: Diastolic blood pressure of 0 mm Hg is impossible in a living patient undergoing a standard screening exam.
  - *Decision*: **Convert 0 to `NaN`.**
* **`SkinThickness`** (227 zeros, 29.56% of dataset):
  - *Physiological Rationale*: Human cutaneous and subcutaneous adipose tissue has measurable non-zero thickness. Zero indicates the caliper test was omitted.
  - *Decision*: **Convert 0 to `NaN`.**
* **`Insulin`** (374 zeros, 48.70% of dataset):
  - *Physiological Rationale*: Serum insulin cannot be 0 in viable human subjects. A total of 374 subjects have zero recorded for 2-hour serum insulin. The specific reason for missing tests is not recorded in the original source documentation, and the missingness mechanism remains unknown.
  - *Decision*: **Convert 0 to `NaN`.**
* **`BMI`** (11 zeros, 1.43% of dataset):
  - *Physiological Rationale*: Body mass index requires positive height and weight. Zero is a clear data collection placeholder.
  - *Decision*: **Convert 0 to `NaN`.**

### Summary of Missingness
| Feature | Zero Count | Missing Percentage | Action Taken |
|---|---|---|---|
| `Pregnancies` | 111 | 0.0% (Valid) | **Preserved as 0** |
| `Glucose` | 5 | 0.65% | Converted to `NaN` |
| `BloodPressure` | 35 | 4.56% | Converted to `NaN` |
| `SkinThickness` | 227 | 29.56% | Converted to `NaN` |
| `Insulin` | 374 | 48.70% | Converted to `NaN` |
| `BMI` | 11 | 1.43% | Converted to `NaN` |
| `DiabetesPedigreeFunction` | 0 | 0.00% | No missing values |
| `Age` | 0 | 0.00% | No missing values |

---

## 5. Imputation Strategy & Data Leakage Prevention

### Why Median Imputation?
1. **Avoid Dropping Rows**: Dropping all rows with at least one missing physiological feature would discard 376 out of 768 records (nearly 49% of the dataset), substantially degrading statistical power.
2. **Robustness to Skewness**: Serum insulin ($\text{skewness} \approx 2.16$) and Skin Thickness exhibit significant right-skewed tails with extreme outliers. Imputing with the **median** is robust to outliers, whereas the mean would artificially inflate imputed values.

### Zero Data Leakage Guarantee
* Imputation values (medians) are **NEVER calculated using the entire dataset**.
* The dataset is split into training (80%) and testing (20%) sets **BEFORE** any imputer is fit.
* The test set is transformed strictly using the training medians.
* **Target Information is Never Used**: Imputation is strictly unsupervised (feature-by-feature medians without conditioning on `Outcome`), ensuring zero target leakage.
