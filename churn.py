import os
import pickle
import joblib
import numpy as np
import streamlit as st

# ═══════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(page_title="📡 Churn Predictor", layout="centered")

# ═══════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#0a0a1a,#1a0a2e,#0d1b2a);
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

/* TITLE */
.title {
    text-align: center;
    font-size: 44px;
    font-weight: 900;
    padding: 12px 0;
    line-height: 1.3;
}

/* ONLY TEXT GRADIENT */
.title span {
    background: linear-gradient(90deg,#00d2ff,#7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.05);
    padding:25px;
    border-radius:20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.6);
}

/* SECTION */
.section {
    margin-top:20px;
    color:#00d2ff;
    font-weight:bold;
}

/* BUTTON */
.stButton>button {
    width:100%;
    padding:14px;
    background: linear-gradient(90deg,#00d2ff,#7b2ff7);
    color:white;
    font-weight:bold;
    border-radius:12px;
}

/* RESULT */
.result-good {
    background: rgba(0,255,150,0.1);
    padding:20px;
    border-radius:15px;
    text-align:center;
    margin-top:20px;
}

.result-bad {
    background: rgba(255,0,0,0.1);
    padding:20px;
    border-radius:15px;
    text-align:center;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════
@st.cache_resource
def load_model():
    if os.path.exists("rf_model.pkl"):
        try:
            return joblib.load("rf_model.pkl")
        except:
            with open("rf_model.pkl", "rb") as f:
                return pickle.load(f)
    else:
        st.error("Model not found")
        st.stop()

model = load_model()

# ═══════════════════════════════════════════════
# HEADER (ICON FIXED)
# ═══════════════════════════════════════════════
st.markdown('''
<h1 class="title">
    📡 <span>CHURN PREDICTOR</span>
</h1>
''', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# CARD
# ═══════════════════════════════════════════════
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="section">👤 Customer Info</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("gender", ["Male", "Female"])
with col2:
    SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
with col3:
    Partner = st.selectbox("Partner", ["Yes", "No"])

col4, col5 = st.columns(2)
with col4:
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
with col5:
    tenure = st.slider("tenure", 0, 72, 12)

st.markdown('<div class="section">📞 Phone</div>', unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
with col7:
    MultipleLines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])

st.markdown('<div class="section">🌐 Internet</div>', unsafe_allow_html=True)

InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])

inet = ["Yes", "No", "No internet service"]

col8, col9 = st.columns(2)
col10, col11 = st.columns(2)
col12, col13 = st.columns(2)

with col8: OnlineSecurity = st.selectbox("OnlineSecurity", inet)
with col9: OnlineBackup = st.selectbox("OnlineBackup", inet)
with col10: DeviceProtection = st.selectbox("DeviceProtection", inet)
with col11: TechSupport = st.selectbox("TechSupport", inet)
with col12: StreamingTV = st.selectbox("StreamingTV", inet)
with col13: StreamingMovies = st.selectbox("StreamingMovies", inet)

st.markdown('<div class="section">💳 Billing</div>', unsafe_allow_html=True)

col14, col15 = st.columns(2)
with col14:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
with col15:
    PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])

PaymentMethod = st.selectbox("PaymentMethod", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

col16, col17 = st.columns(2)
with col16:
    MonthlyCharges = st.number_input("MonthlyCharges", 0.0, 200.0, 65.0)
with col17:
    TotalCharges = st.number_input("TotalCharges", 0.0, 10000.0, 1200.0)

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════
if st.button("🔮 PREDICT CHURN"):

    X = np.array([[SeniorCitizen, tenure, MonthlyCharges, TotalCharges,
    1 if gender=="Female" else 0,1 if gender=="Male" else 0,
    1 if Partner=="No" else 0,1 if Partner=="Yes" else 0,
    1 if Dependents=="No" else 0,1 if Dependents=="Yes" else 0,
    1 if PhoneService=="No" else 0,1 if PhoneService=="Yes" else 0,
    1 if MultipleLines=="No" else 0,1 if MultipleLines=="No phone service" else 0,1 if MultipleLines=="Yes" else 0,
    1 if InternetService=="DSL" else 0,1 if InternetService=="Fiber optic" else 0,1 if InternetService=="No" else 0,
    1 if OnlineSecurity=="No" else 0,1 if OnlineSecurity=="No internet service" else 0,1 if OnlineSecurity=="Yes" else 0,
    1 if OnlineBackup=="No" else 0,1 if OnlineBackup=="No internet service" else 0,1 if OnlineBackup=="Yes" else 0,
    1 if DeviceProtection=="No" else 0,1 if DeviceProtection=="No internet service" else 0,1 if DeviceProtection=="Yes" else 0,
    1 if TechSupport=="No" else 0,1 if TechSupport=="No internet service" else 0,1 if TechSupport=="Yes" else 0,
    1 if StreamingTV=="No" else 0,1 if StreamingTV=="No internet service" else 0,1 if StreamingTV=="Yes" else 0,
    1 if StreamingMovies=="No" else 0,1 if StreamingMovies=="No internet service" else 0,1 if StreamingMovies=="Yes" else 0,
    1 if Contract=="Month-to-month" else 0,1 if Contract=="One year" else 0,1 if Contract=="Two year" else 0,
    1 if PaperlessBilling=="No" else 0,1 if PaperlessBilling=="Yes" else 0,
    1 if PaymentMethod=="Bank transfer (automatic)" else 0,
    1 if PaymentMethod=="Credit card (automatic)" else 0,
    1 if PaymentMethod=="Electronic check" else 0,
    1 if PaymentMethod=="Mailed check" else 0]])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    if pred == 1:
        st.markdown(f'<div class="result-bad"><h2>🚨 CHURN ({prob[1]*100:.2f}%)</h2></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-good"><h2>✅ NO CHURN ({prob[0]*100:.2f}%)</h2></div>', unsafe_allow_html=True)