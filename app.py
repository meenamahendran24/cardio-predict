import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="CardioPredict | Risk assessment",
    page_icon=":material/favorite:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model():
    """Load the selected trained model once per app session."""
    return joblib.load("cardio_logistic_model.joblib")


def risk_level(probability: float) -> tuple[str, str, str]:
    """Return a non-diagnostic risk label and visual treatment."""
    if probability >= 0.70:
        return "Elevated likelihood", "risk-high", "Review with a clinician"
    if probability >= 0.40:
        return "Moderate likelihood", "risk-medium", "Consider a clinical review"
    return "Lower likelihood", "risk-low", "Maintain preventive care"


def make_patient_frame(values: dict) -> pd.DataFrame:
    """Create the exact feature order used during model training."""
    bmi = round(values["weight"] / (values["height"] / 100) ** 2, 2)
    return pd.DataFrame([{
        "gender": 1 if values["gender"] == "Female" else 2,
        "height": values["height"], "weight": values["weight"],
        "ap_hi": values["ap_hi"], "ap_lo": values["ap_lo"],
        "cholesterol": values["cholesterol"], "gluc": values["gluc"],
        "smoke": values["smoke"], "alco": values["alco"],
        "active": values["active"], "age_years": values["age_years"], "bmi": bmi,
    }])


def choice_label(value: int) -> str:
    return "Yes" if value else "No"


model = load_model()

st.markdown("""
<style>
:root { --ink:#132238; --muted:#64748b; --line:#e2e8f0; --canvas:#f7fafc; --brand:#0f766e; --navy:#0f172a; }
.stApp { background:radial-gradient(circle at 6% -8%,rgba(20,184,166,.16),transparent 25rem),radial-gradient(circle at 100% 0%,rgba(56,189,248,.13),transparent 26rem),var(--canvas); color:var(--ink); }
.block-container { max-width:1240px; padding-top:1.35rem; padding-bottom:3.5rem; }
.app-nav { display:flex; align-items:center; justify-content:space-between; padding:.5rem 0 1.3rem; }.brand { display:flex; align-items:center; gap:.65rem; color:var(--navy); font-weight:800; font-size:1.13rem; letter-spacing:-.02em; }.brand-mark { display:grid; place-items:center; height:2.25rem; width:2.25rem; border-radius:.78rem; color:white; background:linear-gradient(135deg,#0f766e,#14b8a6); box-shadow:0 8px 18px rgba(13,148,136,.25); }.nav-note { color:var(--muted); font-size:.84rem; font-weight:600; }
.hero { padding:2.7rem 0 2rem; animation:rise .5s ease-out both; }.eyebrow { color:var(--brand); font-size:.76rem; letter-spacing:.12em; font-weight:800; text-transform:uppercase; margin:0 0 .65rem; }.hero h1 { margin:0; color:var(--navy); font-size:clamp(2rem,4vw,3.4rem); line-height:1.08; letter-spacing:-.045em; max-width:760px; }.hero p { color:var(--muted); font-size:1.06rem; line-height:1.65; margin:1rem 0 0; max-width:650px; }.trust-row { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.35rem; }.trust-pill { border:1px solid #cbd5e1; color:#475569; background:rgba(255,255,255,.76); border-radius:999px; padding:.36rem .72rem; font-size:.78rem; font-weight:650; }
.section-heading { color:var(--navy); font-size:1.2rem; font-weight:750; letter-spacing:-.02em; margin:0; }.section-copy { color:var(--muted); font-size:.9rem; margin:.35rem 0 1rem; } div[data-testid="stForm"] { border:1px solid var(--line); background:rgba(255,255,255,.92); border-radius:1.1rem; padding:1.35rem; box-shadow:0 16px 40px rgba(15,23,42,.06); animation:rise .58s ease-out both; } div[data-testid="stNumberInput"] input,div[data-baseweb="select"]>div { border-radius:.68rem !important; } div[data-testid="stNumberInput"] input:focus,div[data-baseweb="select"]>div:focus-within { border-color:#0d9488 !important; box-shadow:0 0 0 3px rgba(20,184,166,.16) !important; }
.stButton button,div[data-testid="stFormSubmitButton"] button { border-radius:.72rem !important; min-height:2.85rem; font-weight:750; transition:transform .18s ease,box-shadow .18s ease; }.stButton button:hover,div[data-testid="stFormSubmitButton"] button:hover { transform:translateY(-1px); box-shadow:0 9px 18px rgba(15,118,110,.2); }.sidebar-card { border:1px solid var(--line); background:rgba(255,255,255,.84); border-radius:1.1rem; padding:1.35rem; box-shadow:0 12px 30px rgba(15,23,42,.05); animation:rise .65s ease-out both; }.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:.75rem; margin:1rem 0; }.mini-metric { background:#f8fafc; border:1px solid #e8eef5; border-radius:.82rem; padding:.78rem; }.mini-label { color:var(--muted); font-size:.72rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }.mini-value { color:var(--navy); font-size:1.05rem; font-weight:800; margin-top:.18rem; }
.result-card { border-radius:1.1rem; padding:1.35rem; color:var(--navy); animation:rise .35s ease-out both; }.risk-low { background:#ecfdf5; border:1px solid #a7f3d0; }.risk-medium { background:#fffbeb; border:1px solid #fde68a; }.risk-high { background:#fff1f2; border:1px solid #fecdd3; }.result-kicker { color:#475569; font-size:.76rem; letter-spacing:.1em; text-transform:uppercase; font-weight:800; }.result-title { font-size:1.55rem; font-weight:800; letter-spacing:-.035em; margin:.34rem 0; }.result-copy { color:#475569; line-height:1.55; margin:0; }.probability { font-size:2.6rem; font-weight:850; letter-spacing:-.06em; margin:.85rem 0 .45rem; }.disclaimer { border-left:3px solid #14b8a6; color:#475569; font-size:.82rem; line-height:1.5; padding:.1rem 0 .1rem .75rem; margin-top:1.1rem; }
@keyframes rise { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } } @media (max-width:768px) { .block-container { padding:1rem 1rem 2.5rem; }.hero { padding:1.7rem 0; }.nav-note { display:none; } div[data-testid="stForm"] { padding:1rem; } } @media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation-duration:.01ms !important; transition-duration:.01ms !important; } }
</style>
<div class="app-nav"><div class="brand"><span class="brand-mark">&#9829;</span> CardioPredict</div><span class="nav-note">Clinical risk screening workspace</span></div>
<section class="hero"><p class="eyebrow">Machine-learning risk assessment</p><h1>Clearer cardiovascular risk insights.</h1><p>Enter routine health and lifestyle details to receive a transparent, educational probability estimate in seconds.</p><div class="trust-row"><span class="trust-pill">12 clinical & lifestyle signals</span><span class="trust-pill">Logistic Regression model</span><span class="trust-pill">Educational use only</span></div></section>
""", unsafe_allow_html=True)

form_column, summary_column = st.columns([1.55, .85], gap="large")
with form_column:
    st.markdown('<p class="section-heading">Patient profile</p><p class="section-copy">Complete all fields below. Values are checked before prediction.</p>', unsafe_allow_html=True)
    with st.form("risk_assessment_form", border=False):
        st.markdown("##### Personal details")
        personal_left, personal_right = st.columns(2)
        with personal_left:
            age_years = st.number_input("Age (years)", 18, 100, 50, help="Patient age in completed years.")
            height = st.number_input("Height (cm)", 120, 220, 165)
        with personal_right:
            gender_label = st.selectbox("Gender", ["Female", "Male"])
            weight = st.number_input("Weight (kg)", 35.0, 200.0, 70.0, step=.1)
        st.divider(); st.markdown("##### Clinical measurements")
        bp_left, bp_right = st.columns(2)
        with bp_left:
            ap_hi = st.number_input("Systolic blood pressure", 70, 250, 120, help="Upper blood-pressure value in mmHg.")
            cholesterol = st.selectbox("Cholesterol level", [1, 2, 3], format_func=lambda x: {1:"Normal",2:"Above normal",3:"Well above normal"}[x])
        with bp_right:
            ap_lo = st.number_input("Diastolic blood pressure", 40, 150, 80, help="Lower blood-pressure value in mmHg.")
            gluc = st.selectbox("Glucose level", [1, 2, 3], format_func=lambda x: {1:"Normal",2:"Above normal",3:"Well above normal"}[x])
        st.divider(); st.markdown("##### Lifestyle")
        one, two, three = st.columns(3)
        with one: smoke = st.selectbox("Smoking", [0, 1], format_func=choice_label)
        with two: alco = st.selectbox("Alcohol use", [0, 1], format_func=choice_label)
        with three: active = st.selectbox("Physically active", [0, 1], format_func=choice_label)
        submitted = st.form_submit_button("Assess cardiovascular risk", type="primary", icon=":material/analytics:", use_container_width=True)
    if submitted:
        if ap_hi <= ap_lo:
            st.error("Systolic blood pressure must be higher than diastolic blood pressure. Please correct the values and try again.", icon=":material/error:")
        else:
            inputs = {"age_years":age_years,"gender":gender_label,"height":height,"weight":weight,"ap_hi":ap_hi,"ap_lo":ap_lo,"cholesterol":cholesterol,"gluc":gluc,"smoke":smoke,"alco":alco,"active":active}
            with st.spinner("Evaluating the patient profile..."):
                patient_data = make_patient_frame(inputs)
                probability = float(model.predict_proba(patient_data)[0][1])
                prediction = int(model.predict(patient_data)[0])
            st.session_state["assessment"] = {"probability":probability,"prediction":prediction,"bmi":float(patient_data.loc[0,"bmi"])}
with summary_column:
    st.markdown('<div class="sidebar-card"><p class="section-heading">Assessment overview</p><p class="section-copy">The model weighs the same variables used during training.</p><div class="metric-grid"><div class="mini-metric"><div class="mini-label">Test accuracy</div><div class="mini-value">73%</div></div><div class="mini-metric"><div class="mini-label">ROC-AUC</div><div class="mini-value">0.7916</div></div><div class="mini-metric"><div class="mini-label">Dataset</div><div class="mini-value">68,598</div></div><div class="mini-metric"><div class="mini-label">Model</div><div class="mini-value">Logistic</div></div></div><p class="disclaimer">This tool supports learning and discussion. It cannot diagnose disease and must not replace qualified medical advice.</p></div>', unsafe_allow_html=True)
    assessment = st.session_state.get("assessment")
    if assessment:
        label, risk_class, recommendation = risk_level(assessment["probability"])
        st.markdown(f'<div class="result-card {risk_class}"><div class="result-kicker">Latest assessment</div><div class="result-title">{label}</div><p class="result-copy">{recommendation}</p><div class="probability">{assessment["probability"]:.1%}</div><p class="result-copy">Estimated probability &middot; BMI {assessment["bmi"]:.1f}</p></div>', unsafe_allow_html=True)
        if assessment["prediction"] == 1:
            st.warning("The trained model classified this profile as positive. Discuss any health concerns with a qualified healthcare professional.", icon=":material/health_and_safety:")
        else:
            st.success("The trained model classified this profile as negative. This result is not a diagnosis.", icon=":material/check_circle:")
    else:
        st.info("Your result will appear here after you submit the form.", icon=":material/insights:")
