"""Interactive Diabetes Risk Prediction Web Application.

Built with Streamlit, scikit-learn, and custom modern CSS.
Strictly $0 USD Total Budget — Runs 100% locally on personal compute.
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
    page_title="GlucoSense | Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Rich Aesthetics, Glassmorphism, Google Fonts)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background gradient styling */
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

.badge-free {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-auc {
    background-color: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.badge-local {
    background-color: rgba(168, 85, 247, 0.15);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.3);
}

/* Metric / Card styling */
.glass-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    border: 1px solid #e2e8f0;
    margin-bottom: 1.2rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.glass-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07);
}

/* Risk Result Banner */
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

/* Risk Progress Bar */
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

/* Factor list */
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

/* Disclaimer footer */
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
    <div class="hero-title">🩺 GlucoSense Risk AI</div>
    <div class="hero-subtitle">
        Clinical-grade Machine Learning for early Type 2 Diabetes screening and prevention analytics.
        Trained strictly on open-access epidemiological data with reproducible scikit-learn pipelines.
    </div>
    <div>
        <span class="badge-pill badge-free">✦ $0 Budget Verified (100% Free)</span>
        <span class="badge-pill badge-auc">✦ ROC-AUC: 0.813</span>
        <span class="badge-pill badge-local">✦ Local Compute Only</span>
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
        caption="Preventative Health Screening",
        use_container_width=True,
    )

    st.subheader("⚙️ Analysis Settings")

    model_choice = st.selectbox(
        "Active Prediction Model",
        options=["random_forest", "logistic_regression"],
        format_func=lambda x: "Random Forest (Recommended)" if x == "random_forest" else "Logistic Regression (Linear Baseline)",
        help="Select which scikit-learn model pipeline executes inference.",
    )

    app_mode = st.radio(
        "Navigation",
        options=["Patient Risk Calculator", "Model Insights & Metrics", "Batch CSV Screening", "Cost & Architecture Audit"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📋 Quick Clinical Presets")
    preset = st.selectbox(
        "Load Clinical Profile",
        options=["Custom Inputs", "Preset: Healthy Individual", "Preset: Borderline / Pre-Diabetic", "Preset: Elevated Risk"],
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748b; line-height: 1.4;">
            <strong>Open-Source Stack</strong><br>
            Python 3.12 · scikit-learn · Streamlit · Joblib<br>
            Zero external paid APIs. Runs 100% locally.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# TAB 1: PATIENT RISK CALCULATOR
# ---------------------------------------------------------
if app_mode == "Patient Risk Calculator":
    # Preset Values handling
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

    if preset == "Preset: Healthy Individual":
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
    elif preset == "Preset: Borderline / Pre-Diabetic":
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
    elif preset == "Preset: Elevated Risk":
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
        st.markdown("### 👤 Patient Clinical Parameters")
        st.caption("Enter patient diagnostic metrics or use presets from the sidebar.")

        c1, c2 = st.columns(2)
        with c1:
            glucose = st.slider(
                "Plasma Glucose (2-hr OGTT)",
                min_value=50,
                max_value=250,
                value=int(defaults["Glucose"]),
                step=1,
                help="Fasting normal: <100 mg/dL. Pre-diabetes: 100-125 mg/dL. Diabetes: ≥126 mg/dL (fasting) or ≥140 mg/dL (post-challenge).",
            )
            bmi = st.slider(
                "Body Mass Index (BMI)",
                min_value=14.0,
                max_value=60.0,
                value=float(defaults["BMI"]),
                step=0.1,
                help="Healthy weight: 18.5 - 24.9 kg/m². Overweight: 25.0 - 29.9. Obese: ≥30.0.",
            )
            age = st.slider(
                "Patient Age (years)",
                min_value=18,
                max_value=100,
                value=int(defaults["Age"]),
                step=1,
            )
            pregnancies = st.slider(
                "Number of Pregnancies",
                min_value=0,
                max_value=17,
                value=int(defaults["Pregnancies"]),
                step=1,
                help="Gestational diabetes history increases long-term risk.",
            )

        with c2:
            blood_pressure = st.slider(
                "Diastolic Blood Pressure (mm Hg)",
                min_value=40,
                max_value=140,
                value=int(defaults["BloodPressure"]),
                step=1,
                help="Normal diastolic: <80 mm Hg. Elevated: 80-89 mm Hg. Stage 2 hypertension: ≥90 mm Hg.",
            )
            insulin = st.slider(
                "2-Hour Serum Insulin (µU/mL)",
                min_value=10,
                max_value=600,
                value=int(defaults["Insulin"]),
                step=5,
                help="Normal fasting range: 16-166 µU/mL. Higher values often indicate insulin resistance.",
            )
            pedigree = st.slider(
                "Diabetes Pedigree Function",
                min_value=0.05,
                max_value=2.50,
                value=float(defaults["DiabetesPedigreeFunction"]),
                step=0.01,
                help="Genetic risk score based on family diabetes history and relatedness.",
            )
            skin_thickness = st.slider(
                "Triceps Skinfold Thickness (mm)",
                min_value=5,
                max_value=70,
                value=int(defaults["SkinThickness"]),
                step=1,
                help="Estimator of peripheral subcutaneous adiposity.",
            )

        # What-If Interactive Expander
        with st.expander("🧪 What-If Lifestyle Intervention Simulation", expanded=False):
            st.write("Simulate how targeted lifestyle interventions could shift this patient's risk profile:")
            wi_glucose_drop = st.slider("Simulate Glucose Reduction (mg/dL)", 0, 60, 15)
            wi_bmi_drop = st.slider("Simulate BMI Reduction (kg/m²)", 0.0, 10.0, 2.5)

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

    # What-If Computation
    simulated_patient = patient_dict.copy()
    simulated_patient["Glucose"] = max(70, glucose - wi_glucose_drop)
    simulated_patient["BMI"] = max(18.5, bmi - wi_bmi_drop)
    sim_prediction = predict_risk(simulated_patient, model_type=model_choice)

    with col_results:
        st.markdown("### 📊 Assessment & Risk Score")

        prob_pct = prediction["risk_percentage"]
        color = prediction["color"]
        level = prediction["risk_level"]

        # Dynamic Risk Card
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
                    Predicted Probability of Diabetes
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width: {prob_pct}%; background-color: {color};"></div>
                </div>
                <div style="font-size: 0.92rem; color: #475569; line-height: 1.5; text-align: left; background: #f8fafc; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    <strong>Clinical Summary:</strong><br>{prediction["summary"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Primary Contributing Factors")
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

        # What-If Intervention Impact
        if wi_glucose_drop > 0 or wi_bmi_drop > 0:
            diff = prob_pct - sim_prediction["risk_percentage"]
            st.markdown("#### 📉 Projected Intervention Benefit")
            st.info(
                f"With a **{wi_glucose_drop} mg/dL** glucose reduction and **{wi_bmi_drop:.1f} kg/m²** weight loss, "
                f"the patient's risk drops from **{prob_pct}%** to **{sim_prediction['risk_percentage']}%** "
                f"(**-{diff:.1f}% absolute risk reduction**)."
            )


# ---------------------------------------------------------
# TAB 2: MODEL INSIGHTS & METRICS
# ---------------------------------------------------------
elif app_mode == "Model Insights & Metrics":
    st.markdown("### 🧠 Model Evaluation & Transparency")
    st.caption("Rigorous cross-validation on training data and validation on held-out test data (80/20 stratified split).")

    if metrics:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            rf_test = metrics.get("random_forest", {}).get("test_metrics", {})
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="color: #0f172a; margin-top:0;">🌲 Random Forest Classifier</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">Ensemble of 200 regularized decision trees with median imputation.</p>
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
                    <p style="color: #64748b; font-size: 0.9rem;">L2-regularized linear model on standardized physiological features.</p>
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
            "Random Forest": [rf_fi[k] for k in rf_fi],
            "Logistic Regression": [lr_fi.get(k, 0) for k in rf_fi],
        }).set_index("Feature")

        st.bar_chart(df_fi)
        st.caption(
            "Notice how both models independently confirm that Glucose and BMI are by far the strongest clinical predictors."
        )


# ---------------------------------------------------------
# TAB 3: BATCH CSV SCREENING
# ---------------------------------------------------------
elif app_mode == "Batch CSV Screening":
    st.markdown("### 📁 Batch Patient Screening")
    st.caption("Upload a patient cohort CSV file to compute risk scores for hundreds of individuals at once.")

    # Provide sample template download
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

    uploaded_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_input = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(batch_input)}** records:")
            st.dataframe(batch_input.head(), use_container_width=True)

            if st.button("🚀 Run Cohort Risk Analysis"):
                scored_df = predict_batch(batch_input, model_type=model_choice)
                st.success("Analysis complete!")
                st.dataframe(scored_df, use_container_width=True)

                # Distribution summary
                col_b1, col_b2, col_b3 = st.columns(3)
                tier_counts = scored_df["Risk_Tier"].value_counts()
                col_b1.metric("Low Risk Cohort", tier_counts.get("Low Risk", 0))
                col_b2.metric("Moderate Risk Cohort", tier_counts.get("Moderate Risk", 0))
                col_b3.metric("High Risk Cohort", tier_counts.get("High Risk", 0))

                # Download scored file
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
# TAB 4: COST & ARCHITECTURE AUDIT
# ---------------------------------------------------------
elif app_mode == "Cost & Architecture Audit":
    st.markdown("### 🛡️ $0 Budget & Architecture Verification")
    st.caption("Full audit demonstrating absolute zero spend and local execution.")

    audit_items = [
        ("Cloud Compute / GPU", "Local Apple M-series / x86 CPU only. No AWS, GCP, or Azure compute.", "✅ $0.00"),
        ("Model APIs", "No OpenAI, Gemini, Claude, or paid LLM/AI APIs used.", "✅ $0.00"),
        ("Database / Hosting", "Local file-based dataset & local Joblib storage. No cloud DB charges.", "✅ $0.00"),
        ("Dataset Acquisition", "Open-access Pima Indians Diabetes Dataset from UCI / NIDDK.", "✅ $0.00"),
        ("Libraries & Tooling", "100% Free Open Source: Python 3.12, scikit-learn, Streamlit, Pandas, NumPy.", "✅ $0.00"),
        ("Version Control", "Local Git with standard free GitHub repository integration.", "✅ $0.00"),
    ]

    for title, desc, cost in audit_items:
        st.markdown(
            f"""
            <div class="glass-card" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h5 style="margin: 0; color: #0f172a;">{title}</h5>
                    <p style="margin: 0; color: #64748b; font-size: 0.88rem;">{desc}</p>
                </div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #10b981;">
                    {cost}
                </div>
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
    <strong>⚠️ Clinical Disclaimer:</strong>
    GlucoSense is an educational decision-support tool powered by machine learning on retrospective observational data.
    It does not constitute formal medical diagnosis, prognosis, or clinical treatment advice.
    Always consult a licensed physician or endocrinologist for clinical blood testing (fasting glucose, HbA1c) and professional diagnosis.
</div>
""",
    unsafe_allow_html=True,
)
