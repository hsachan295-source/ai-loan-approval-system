import streamlit as st
import numpy as np
import pickle
import os

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanSense · Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp { background: #0a0d14; color: #e2e8f0; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem; max-width: 1100px; }

.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.5rem 0; letter-spacing: -1px;
}
.hero-subtitle { color: #64748b; font-size: 0.95rem; font-weight: 300; margin: 0; }
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8; font-family: 'Space Mono', monospace; font-size: 0.7rem;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 1rem;
    letter-spacing: 2px; text-transform: uppercase;
}
.section-header {
    font-family: 'Space Mono', monospace; font-size: 0.75rem;
    letter-spacing: 3px; text-transform: uppercase; color: #38bdf8;
    margin: 2rem 0 1.2rem; display: flex; align-items: center; gap: 10px;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}
.stNumberInput > div > div > input, .stSelectbox > div > div {
    background: #0a0d14 !important; border: 1px solid #1e293b !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important; font-size: 0.9rem !important;
}
.stNumberInput label, .stSelectbox label {
    color: #94a3b8 !important; font-size: 0.82rem !important;
    font-weight: 500 !important; letter-spacing: 0.3px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.9rem 3rem !important; font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important; font-weight: 700 !important;
    letter-spacing: 1px !important; text-transform: uppercase !important;
    width: 100% !important; box-shadow: 0 4px 20px rgba(14,165,233,0.25) !important;
    margin-top: 1rem !important;
}
.stat-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.stat-card {
    flex: 1; background: #0f172a; border: 1px solid #1e293b;
    border-radius: 14px; padding: 1.2rem 1.5rem; text-align: center;
}
.stat-value {
    font-family: 'Space Mono', monospace; font-size: 1.5rem;
    font-weight: 700; color: #38bdf8; display: block;
}
.stat-label { font-size: 0.75rem; color: #475569; letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }
hr { border: none; border-top: 1px solid #1e293b; margin: 1.5rem 0; }
[data-testid="column"] { padding: 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_model.pkl")
        return pickle.load(open(model_path, "rb"))
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model = load_model()

# ── Hero Banner ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">⬡ AI-Powered Risk Engine</div>
    <h1 class="hero-title">LoanSense</h1>
    <p class="hero-subtitle">Gaussian Naive Bayes · 84.97% Accuracy · Real-time Credit Risk Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
    <div class="stat-card"><span class="stat-value">84.97%</span><span class="stat-label">Model Accuracy</span></div>
    <div class="stat-card"><span class="stat-value">22</span><span class="stat-label">Input Features</span></div>
    <div class="stat-card"><span class="stat-value">GNB</span><span class="stat-label">Algorithm</span></div>
    <div class="stat-card"><span class="stat-value">Live</span><span class="stat-label">Prediction Mode</span></div>
</div>
""", unsafe_allow_html=True)

# ── Section 01: Financial Profile ───────────────────────────
st.markdown('<div class="section-header">01 · Financial Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    income       = st.number_input("Annual Income (₹)", min_value=0, max_value=10000000, value=500000, step=10000)
    loan_amt     = st.number_input("Loan Amount (₹)",   min_value=0, max_value=10000000, value=200000, step=5000)
    interest     = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
with col2:
    percent_income = st.number_input("Loan as % of Income",    min_value=0.0, max_value=100.0, value=20.0, step=0.5)
    credit_score   = st.number_input("Credit Score (300–900)", min_value=300,  max_value=900,   value=650,  step=1)
    credit_hist    = st.number_input("Credit History (Years)", min_value=0,    max_value=50,    value=5,    step=1)

# ── Section 02: Personal Details ────────────────────────────
st.markdown('<div class="section-header">02 · Personal Details</div>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    age        = st.number_input("Age",                       min_value=18, max_value=100, value=30, step=1)
    experience = st.number_input("Work Experience (Years)",   min_value=0,  max_value=50,  value=5,  step=1)
with col4:
    education  = st.selectbox("Education Level",    ["Associate", "Bachelor", "Doctorate", "High School", "Master"])
    home       = st.selectbox("Home Ownership",     ["MORTGAGE", "OTHER", "OWN", "RENT"])
with col5:
    prev_default = st.selectbox("Previous Default History", ["No", "Yes"])

# ── Section 03: Loan Intent ──────────────────────────────────
st.markdown('<div class="section-header">03 · Loan Intent</div>', unsafe_allow_html=True)

intent = st.selectbox(
    "What is the purpose of this loan?",
    ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"],
    format_func=lambda x: {
        "DEBTCONSOLIDATION": "🔄 Debt Consolidation",
        "EDUCATION":         "🎓 Education",
        "HOMEIMPROVEMENT":   "🏠 Home Improvement",
        "MEDICAL":           "🏥 Medical",
        "PERSONAL":          "👤 Personal",
        "VENTURE":           "🚀 Venture / Business"
    }[x]
)

# ── Feature Builder — exactly 22 features ───────────────────
def build_features():
    features = [
        age,                                         # person_age
        income,                                      # person_income
        experience,                                  # person_emp_exp
        loan_amt,                                    # loan_amnt
        interest,                                    # loan_int_rate
        percent_income,                              # loan_percent_income
        credit_hist,                                 # cb_person_cred_hist_length
        credit_score,                                # credit_score
        0,                                           # person_gender_male (neutral)
        1 if education == "Bachelor"    else 0,      # person_education_Bachelor
        1 if education == "Doctorate"   else 0,      # person_education_Doctorate
        1 if education == "High School" else 0,      # person_education_High School
        1 if education == "Master"      else 0,      # person_education_Master
        1 if home == "OTHER"            else 0,      # person_home_ownership_OTHER
        1 if home == "OWN"              else 0,      # person_home_ownership_OWN
        1 if home == "RENT"             else 0,      # person_home_ownership_RENT
        1 if intent == "EDUCATION"      else 0,      # loan_intent_EDUCATION
        1 if intent == "HOMEIMPROVEMENT" else 0,     # loan_intent_HOMEIMPROVEMENT
        1 if intent == "MEDICAL"        else 0,      # loan_intent_MEDICAL
        1 if intent == "PERSONAL"       else 0,      # loan_intent_PERSONAL
        1 if intent == "VENTURE"        else 0,      # loan_intent_VENTURE
        1 if prev_default == "Yes"      else 0,      # previous_loan_defaults_on_file_Yes
    ]
    return np.array([features])

# ── Predict Button ───────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    predict_clicked = st.button("⬡ Run Risk Analysis")

if predict_clicked:
    if model is None:
        st.error("⚠️ Model file not found. Please ensure loan_model.pkl is in the project root.")
    else:
        data = build_features()
        try:
            prediction = model.predict(data)
            st.markdown("<br>", unsafe_allow_html=True)
            col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
            with col_r2:
                if prediction[0] == 0:
                    st.success("✅ LOAN APPROVED — Low Default Risk Detected")
                    st.markdown("<div style='text-align:center;color:#475569;font-size:0.82rem;margin-top:0.5rem;'>Applicant profile appears creditworthy based on 22 features analyzed.</div>", unsafe_allow_html=True)
                else:
                    st.error("❌ HIGH DEFAULT RISK — Loan Not Recommended")
                    st.markdown("<div style='text-align:center;color:#475569;font-size:0.82rem;margin-top:0.5rem;'>Significant risk factors detected. Manual review advised.</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#334155;font-size:0.75rem;margin-top:3rem;
font-family:"Space Mono",monospace;letter-spacing:1px;'>
    LOANSENSE · GAUSSIAN NAIVE BAYES · FOR EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)