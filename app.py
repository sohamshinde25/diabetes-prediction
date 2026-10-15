"""Interactive Diabetes Risk Prediction Web Application.

Built with Streamlit, scikit-learn, and custom modern CSS.
Exploratory machine learning application on the historical Pima Indians Diabetes dataset.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

from src.predict import predict_risk, predict_batch, load_model
from src.data_loader import FEATURE_RANGES, EXPECTED_COLUMNS
from src.preprocessing import DEFAULT_FEATURE_NAMES

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction | ML Explorer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Clean Modern Aesthetics, Glassmorphism, Google Fonts)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main {
    background: radial-gradient(circle at top left, #f8fafc 0%, #edf2f7 100%);
}

/* Hero Section */
.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: white;
    padding: 2.2rem 2.5rem;
    border-radius: 18px;
    box-shadow: 0 20px 25px -5px rgba(15, 23, 42, 0.25), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(90deg, #ffffff 0%, #93c5fd 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}

.badge-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.6rem;
    letter-spacing: 0.02em;
}

.badge-auc {
    background-color: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.badge-local {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.glass-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    border: 1px solid #e2e8f0;
    margin-bottom: 1.2rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.risk-meter-container {
    background: white;
    border-radius: 16px;
    padding: 1.8rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    text-align: center;
}

.risk-level-tag {
    display: inline-block;
    padding: 0.45rem 1.4rem;
    border-radius: 9999px;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}

.risk-percentage-val {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0.5rem 0;
}

.progress-bg {
    background: #e2e8f0;
    border-radius: 9999px;
    height: 14px;
    overflow: hidden;
    margin: 1.2rem 0;
}

.progress-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.6s ease-in-out;
}

.factor-item {
    display: flex;
    align-items: center;
    background: #f8fafc;
    border-left: 4px solid #ef4444;
    padding: 0.65rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.5rem;
    font-size: 0.92rem;
    color: #334155;
}

.factor-item-ok {
    border-left-color: #10b981;
}

.disclaimer-card {
    background: #f1f5f9;
    border-left: 4px solid #64748b;
    padding: 0.9rem 1.2rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem;
    color: #475569;
    margin-top: 2rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def get_metrics():
    """Load model training evaluation metrics."""
    metrics_path = Path(__file__).resolve().parent / "models" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}


# Hero Header Section
st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">🩺 Diabetes Risk Prediction</div>
    <div class="hero-subtitle">
        An interactive machine learning exploration evaluating Logistic Regression and Random Forest
        classifiers on the historical Pima Indians Diabetes benchmark dataset.
    </div>
    <div>
        <span class="badge-pill badge-auc">✦ ROC-AUC: 0.812 (RF) / 0.813 (LR)</span>
        <span class="badge-pill badge-local">✦ Leakage-Aware scikit-learn Pipeline</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

metrics = get_metrics()

# Sidebar Navigation & Settings
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=600&auto=format&fit=crop&q=80",
        caption="Health Screening Analytics",
        use_container_width=True,
    )

    st.subheader("⚙️ Model Configuration")

    model_choice = st.selectbox(
        "Active Prediction Pipeline",
        options=["random_forest", "logistic_regression"],
        format_func=lambda x: "Random Forest (Non-linear Ensemble)" if x == "random_forest" else "Logistic Regression (Linear Baseline)",
        help="Select which scikit-learn model pipeline executes inference.",
    )

    app_mode = st.radio(
        "Navigation",
        options=["Patient Risk Calculator", "Model Insights & Metrics", "Batch CSV Screening", "Dataset & Architecture Details"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📋 Sample Input Profiles")
    preset = st.selectbox(
        "Load Benchmark Profile",
        options=["Custom Inputs", "Preset: Normative Profile", "Preset: Intermediate Profile", "Preset: Elevated Risk Profile"],
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748b; line-height: 1.4;">
            <strong>Technical Stack</strong><br>
            Python 3.12 · scikit-learn · Streamlit · Joblib<br>
            Local execution with open-source libraries.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# TAB 1: PATIENT RISK CALCULATOR
# ---------------------------------------------------------
if app_mode == "Patient Risk Calculator":
    defaults = {
        "Pregnancies": 1,
        "Glucose": 105,
        "BloodPressure": 72,
        "SkinThickness": 24,
        "Insulin": 85,
        "BMI": 26.5,
        "DiabetesPedigreeFunction": 0.35,
        "Age": 32,
    }

    if preset == "Preset: Normative Profile":
        defaults = {
            "Pregnancies": 0,
            "Glucose": 88,
            "BloodPressure": 68,
            "SkinThickness": 19,
            "Insulin": 65,
            "BMI": 22.4,
            "DiabetesPedigreeFunction": 0.20,
            "Age": 26,
        }
    elif preset == "Preset: Intermediate Profile":
        defaults = {
            "Pregnancies": 2,
            "Glucose": 125,
            "BloodPressure": 78,
            "SkinThickness": 28,
            "Insulin": 130,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.55,
            "Age": 42,
        }
    elif preset == "Preset: Elevated Risk Profile":
        defaults = {
            "Pregnancies": 4,
            "Glucose": 172,
            "BloodPressure": 86,
            "SkinThickness": 38,
            "Insulin": 220,
            "BMI": 36.2,
            "DiabetesPedigreeFunction": 0.95,
            "Age": 54,
        }

    col_inputs, col_results = st.columns([1.15, 1], gap="large")

    with col_inputs:
        st.markdown("### 👤 Patient Input Features")
        st.caption("Adjust input parameters or select a reference profile from the sidebar.")

        c1, c2 = st.columns(2)
        with c1:
            glucose = st.slider(
                "Plasma Glucose (2-hr OGTT, mg/dL)",
                min_value=50,
                max_value=250,
                value=int(defaults["Glucose"]),
                step=1,
                help="Reference threshold: <100 mg/dL normal, 100-125 borderline, >=140 elevated post-challenge.",
            )
            bmi = st.slider(
                "Body Mass Index (BMI, kg/m²)",
                min_value=14.0,
                max_value=60.0,
                value=float(defaults["BMI"]),
                step=0.1,
                help="Reference categories: 18.5-24.9 normal, 25.0-29.9 overweight, >=30.0 obese.",
            )
            age = st.slider(
                "Age (years)",
                min_value=21,
                max_value=100,
                value=max(21, int(defaults["Age"])),
                step=1,
                help="Cohort inclusion criterion in original study was age >= 21.",
            )
            pregnancies = st.slider(
                "Number of Pregnancies",
                min_value=0,
                max_value=17,
                value=int(defaults["Pregnancies"]),
                step=1,
                help="Zero is biologically valid for nulliparous women.",
            )

        with c2:
            blood_pressure = st.slider(
                "Diastolic Blood Pressure (mm Hg)",
                min_value=40,
                max_value=140,
                value=int(defaults["BloodPressure"]),
                step=1,
            )
            insulin = st.slider(
                "2-Hour Serum Insulin (µU/mL)",
                min_value=10,
                max_value=600,
                value=int(defaults["Insulin"]),
                step=5,
            )
            pedigree = st.slider(
                "Diabetes Pedigree Function",
                min_value=0.05,
                max_value=2.50,
                value=float(defaults["DiabetesPedigreeFunction"]),
                step=0.01,
                help="Continuous genetic risk score based on family history.",
            )
            skin_thickness = st.slider(
                "Triceps Skinfold Thickness (mm)",
                min_value=5,
                max_value=70,
                value=int(defaults["SkinThickness"]),
                step=1,
            )

        # Model Sensitivity Exploration
        with st.expander("🧪 Model Sensitivity Exploration", expanded=False):
            st.caption(
                "Observe how adjusting features shifts the model's estimated probability across the learned response surface. "
                "Note: Associative ML models capture statistical correlations, not causal intervention effects."
            )
            wi_glucose_delta = st.slider("Perturb Glucose (mg/dL)", -40, 40, -15)
            wi_bmi_delta = st.slider("Perturb BMI (kg/m²)", -8.0, 8.0, -2.5)

    # Compute Prediction
    patient_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }

    prediction = predict_risk(patient_dict, model_type=model_choice)

    # Sensitivity Computation
    simulated_patient = patient_dict.copy()
    simulated_patient["Glucose"] = max(50, glucose + wi_glucose_delta)
    simulated_patient["BMI"] = max(14.0, bmi + wi_bmi_delta)
    sim_prediction = predict_risk(simulated_patient, model_type=model_choice)

    with col_results:
        st.markdown("### 📊 Model Output & Probability")

        prob_pct = prediction["risk_percentage"]
        color = prediction["color"]
        level = prediction["risk_level"]

        st.markdown(
            f"""
            <div class="risk-meter-container">
                <span class="risk-level-tag" style="background-color: {color}22; color: {color}; border: 1px solid {color}55;">
                    ● {level}
                </span>
                <div class="risk-percentage-val" style="color: {color};">
                    {prob_pct}%
                </div>
                <div style="font-size: 0.95rem; color: #64748b; font-weight: 500;">
                    Model Predicted Probability
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: {prob_pct}%; background-color: {color};"></div>
                </div>
                <div style="font-size: 0.92rem; color: #475569; line-height: 1.5; text-align: left; background: #f8fafc; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    <strong>Summary:</strong><br>{prediction["summary"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Notable Contributing Indicators")
        if prediction["contributing_factors"]:
            for factor in prediction["contributing_factors"]:
                st.markdown(
                    f"""<div class="factor-item">⚠️ &nbsp; {factor}</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """<div class="factor-item factor-item-ok">✅ &nbsp; All physiological indicators are within normative ranges.</div>""",
                unsafe_allow_html=True,
            )

        # Sensitivity Impact Display
        if wi_glucose_delta != 0 or wi_bmi_delta != 0:
            diff = sim_prediction["risk_percentage"] - prob_pct
            st.markdown("#### 📈 Response Surface Shift")
            st.info(
                f"With Glucose adjusted by **{wi_glucose_delta:+d} mg/dL** and BMI by **{wi_bmi_delta:+.1f} kg/m²**, "
                f"the model output shifts from **{prob_pct}%** to **{sim_prediction['risk_percentage']}%** "
                f"(**{diff:+.1f}% probability shift**)."
            )


# ---------------------------------------------------------
# TAB 2: MODEL INSIGHTS & METRICS
# ---------------------------------------------------------
elif app_mode == "Model Insights & Metrics":
    st.markdown("### 🧠 Model Evaluation & Comparison")
    st.caption("Evaluated on held-out test data (80/20 stratified split, N=154).")

    if metrics:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            rf_test = metrics.get("random_forest", {}).get("test_metrics", {})
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="color: #0f172a; margin-top:0;">🌲 Random Forest Classifier</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">Ensemble of 200 regularized trees with median imputation (scale-invariant).</p>
                    <table style="width: 100%; font-size: 0.95rem; border-collapse: collapse;">
                        <tr><td style="padding: 6px 0; color:#475569;">Test Accuracy</td><td style="text-align: right; font-weight:700;">{rf_test.get('accuracy', 0)*100:.1f}%</td></tr>
                        <tr><td style="padding: 6px 0; color:#475569;">Test ROC-AUC</td><td style="text-align: right; font-weight:700; color: #10b981;">{rf_test.get('roc_auc', 0):.3f}</td></tr>
                        <tr><td style="padding: 6px 0; color:#475569;">Test Precision</td><td style="text-align: right; font-weight:700;">{rf_test.get('precision', 0)*100:.1f}%</td></tr>
                        <tr><td style="padding: 6px 0; color:#475569;">Test Recall (Sensitivity)</td><td style="text-align: right; font-weight:700;">{rf_test.get('recall', 0)*100:.1f}%</td></tr>
                        <tr><td style="padding: 6px 0; color:#475569;">Test Specificity</td><td style="text-align: right; font-weight:700;">{rf_test.get('specificity', 0)*100:.1f}%</td></tr>
                        <tr><td style="padding: 6px 0; color:#475569;">Test F1-Score</td><td style="text-align: right; font-weight:700;">{rf_test.get('f1', 0):.3f}</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m2:
            lr_test = metrics.get("logistic_regression", {}).get("test_metrics", {})
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="color: #0f172a; margin-top:0;">📈 Logistic Regression (Baseline)</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">L2-regularized linear model with median imputation and standard scaling.</p>
                    <table style="width: 100%; font-size: 0.95rem; border-collapse: collapse;">
                        <tr><td style="padding: 6px 0; color:#475569;">Test Accuracy</td><td style="text-align: right; font-weight:700;">{lr_test.get('accuracy', 0)*100:.1f}%</td></tr>
                        <tr><td style="padding: 6px 0; color:#475569;">Test ROC-AUC</td><td style="text-align: right; font-weight:700; color: #3b82f6;">{lr_test.get('roc_auc', 0):.3f}</td></tr>
                        <tr><td style="padding: 6px 0; color:#475569;">Test Precision</td><td style="text-align: right; font-weight:700;">{lr_test.get('precision', 0)*100:.1f}%</td></tr>
                        <tr><td style="padding: 6px 0; color:#475569;">Test Recall (Sensitivity)</td><td style="text-align: right; font-weight:700;">{lr_test.get('recall', 0)*100:.1f}%</td></tr>
                        <tr><td style="padding: 6px 0; color:#475569;">Test Specificity</td><td style="text-align: right; font-weight:700;">{lr_test.get('specificity', 0)*100:.1f}%</td></tr>
                        <tr><td style="padding: 6px 0; color:#475569;">Test F1-Score</td><td style="text-align: right; font-weight:700;">{lr_test.get('f1', 0):.3f}</td></tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Feature Importance Chart
        st.markdown("#### ⚖️ Relative Feature Importance Comparison")
        rf_fi = metrics.get("random_forest", {}).get("feature_importance", {})
        lr_fi = metrics.get("logistic_regression", {}).get("feature_importance", {})

        df_fi = pd.DataFrame({
            "Feature": list(rf_fi.keys()),
            "Random Forest (MDI)": [rf_fi[k] for k in rf_fi],
            "Logistic Regression (Norm |Coef|)": [lr_fi.get(k, 0) for k in rf_fi],
        }).set_index("Feature")

        st.bar_chart(df_fi)
        st.caption(
            "Feature importance reflects statistical association in the training data, not causal clinical evidence."
        )


# ---------------------------------------------------------
# TAB 3: BATCH CSV SCREENING
# ---------------------------------------------------------
elif app_mode == "Batch CSV Screening":
    st.markdown("### 📁 Batch Cohort Screening")
    st.caption("Upload a patient cohort CSV file to compute model estimates for multiple records.")

    sample_df = pd.DataFrame([
        {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": 94, "BMI": 26.6, "DiabetesPedigreeFunction": 0.35, "Age": 31},
        {"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.63, "Age": 50},
        {"Pregnancies": 8, "Glucose": 183, "BloodPressure": 64, "SkinThickness": 0, "Insulin": 0, "BMI": 23.3, "DiabetesPedigreeFunction": 0.67, "Age": 32},
    ])
    csv_sample = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Sample CSV Template",
        data=csv_sample,
        file_name="diabetes_screening_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Cohort CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_input = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_input)}** records:")
            st.dataframe(batch_input.head(), use_container_width=True)

            if st.button("🚀 Run Cohort Risk Analysis"):
                scored_df = predict_batch(batch_input, model_type=model_choice)
                st.success("Analysis complete!")
                st.dataframe(scored_df, use_container_width=True)

                col_b1, col_b2, col_b3 = st.columns(3)
                tier_counts = scored_df["Risk_Tier"].value_counts()
                col_b1.metric("Low Risk Cohort", tier_counts.get("Low Risk", 0))
                col_b2.metric("Moderate Risk Cohort", tier_counts.get("Moderate Risk", 0))
                col_b3.metric("High Risk Cohort", tier_counts.get("High Risk", 0))

                csv_out = scored_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "💾 Download Scored Results (CSV)",
                    data=csv_out,
                    file_name="diabetes_risk_results.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")


# ---------------------------------------------------------
# TAB 4: DATASET & ARCHITECTURE DETAILS
# ---------------------------------------------------------
elif app_mode == "Dataset & Architecture Details":
    st.markdown("### 🔬 Dataset & Architecture Overview")

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color: #0f172a; margin-top:0;">Dataset Provenance & Cohort</h4>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.6;">
                <strong>Source:</strong> National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), 1988.<br>
                <strong>Cohort:</strong> 768 female subjects aged &ge; 21 of Pima Indian heritage living near Phoenix, Arizona.<br>
                <strong>Missing Value Handling:</strong> Biologically impossible zeros in Glucose (5), BloodPressure (35), SkinThickness (227),
                Insulin (374), and BMI (11) are converted to NaN and imputed using training-set medians. Legitimate zeros in Pregnancies (111) are strictly preserved.<br>
                <strong>Generalizability Limitation:</strong> Findings from this historical cohort cannot be assumed to generalize across diverse clinical populations without external validation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🔬 Reproducibility Commands")
    st.code(
        """
# 1. Run automated test suite
pytest -v

# 2. Retrain models locally
python src/model_trainer.py

# 3. Launch interactive web dashboard
streamlit run app.py
        """,
        language="bash",
    )


# Medical Disclaimer
st.markdown(
    """
<div class="disclaimer-card">
    <strong>⚠️ Educational & Research Disclaimer:</strong>
    This application is an educational machine learning demonstration developed for exploratory analysis.
    It does not constitute formal medical diagnosis, clinical prognosis, or treatment advice.
    Always consult a licensed medical professional for clinical evaluation and laboratory testing.
</div>
""",
    unsafe_allow_html=True,
)
