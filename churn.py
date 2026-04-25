import os
import pickle
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stAppViewContainer"] { background: #F4F3F0; }
[data-testid="stHeader"] { background: transparent; }

.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 3rem; }

h1 { font-size: 24px !important; font-weight: 600 !important; color: #1a1a1a !important; letter-spacing: -0.3px; }

.card {
    background: #ffffff;
    border: 0.5px solid rgba(0,0,0,0.09);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 14px;
}

.section-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #999;
    margin-bottom: 14px;
}

.result-wrap {
    border-radius: 16px;
    padding: 28px 20px 22px;
    text-align: center;
    margin-bottom: 14px;
}
.result-churn  { background: #FEF0F0; border: 1px solid #F09595; }
.result-safe   { background: #EAF3DE; border: 1px solid #97C459; }
.result-neutral{ background: #F4F3F0; border: 1px solid rgba(0,0,0,0.1); }

.result-pct { font-size: 64px; font-weight: 600; letter-spacing: -3px; line-height: 1; }
.result-churn   .result-pct { color: #A32D2D; }
.result-safe    .result-pct { color: #3B6D11; }
.result-neutral .result-pct { color: #bbb; }

.result-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.9px; font-weight: 500; margin-top: 8px; }
.result-churn   .result-label { color: #993535; }
.result-safe    .result-label { color: #3B6D11; }
.result-neutral .result-label { color: #bbb; }

.result-desc { font-size: 13px; color: #666; margin-top: 12px; line-height: 1.6; }

.factor { margin-bottom: 11px; }
.factor-top { display: flex; justify-content: space-between; font-size: 12px; color: #555; margin-bottom: 5px; }
.factor-bg   { background: #EDECEA; border-radius: 99px; height: 5px; }
.factor-fill { height: 100%; border-radius: 99px; }

[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: rgba(0,0,0,0.11) !important;
    background: #FAFAF9 !important;
    font-size: 13px !important;
}
[data-baseweb="select"] > div:hover { border-color: #5340D8 !important; }

input[type="number"] {
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.11) !important;
    background: #FAFAF9 !important;
    font-size: 13px !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #5340D8 !important;
    border-color: #5340D8 !important;
}

.stButton > button {
    background: #5340D8 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 24px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    width: 100%;
    letter-spacing: 0.1px;
}
.stButton > button:hover { opacity: 0.88; }

label { font-size: 12px !important; color: #666 !important; font-weight: 400 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("rf_model.pkl"):
        try:
            return joblib.load("rf_model.pkl")
        except Exception:
            with open("rf_model.pkl", "rb") as f:
                return pickle.load(f)
    st.error("rf_model.pkl not found. Place it in the same folder as app.py.")
    st.stop()

model = load_model()

# ── Header ─────────────────────────────────────────────
st.markdown("<h1>Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#999;font-size:13px;margin-top:-10px;margin-bottom:20px'>Fill in customer details — result appears on the right.</p>", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

# ══════════════════════════════════════════════════════
# LEFT — Form
# ══════════════════════════════════════════════════════
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Customer info</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: gender  = st.selectbox("Gender",         ["Male", "Female"])
    with c2: senior  = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with c3: partner = st.selectbox("Partner",        ["Yes", "No"])
    c4, c5 = st.columns(2)
    with c4: dep    = st.selectbox("Dependents", ["Yes", "No"])
    with c5: tenure = st.slider("Tenure (months)", 0, 72, 12)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Phone</div>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    with c6: phone     = st.selectbox("Phone service",  ["Yes", "No"])
    with c7: multiline = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Internet</div>', unsafe_allow_html=True)
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    opts = ["Yes", "No", "No internet service"]
    c8,  c9  = st.columns(2)
    c10, c11 = st.columns(2)
    c12, c13 = st.columns(2)
    with c8:  sec = st.selectbox("Online security",   opts)
    with c9:  bak = st.selectbox("Online backup",     opts)
    with c10: dev = st.selectbox("Device protection", opts)
    with c11: tec = st.selectbox("Tech support",      opts)
    with c12: tv  = st.selectbox("Streaming TV",      opts)
    with c13: mov = st.selectbox("Streaming movies",  opts)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Billing</div>', unsafe_allow_html=True)
    c14, c15 = st.columns(2)
    with c14: contract  = st.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
    with c15: paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    c16, c17 = st.columns(2)
    with c16: monthly = st.number_input("Monthly charges ($)",  0.0, 200.0,   65.0, step=0.01)
    with c17: total   = st.number_input("Total charges ($)",    0.0, 10000.0, 1200.0, step=0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    predict = st.button("Predict churn →")

# ══════════════════════════════════════════════════════
# RIGHT — Result
# ══════════════════════════════════════════════════════
with right:
    if predict:
        X = np.array([[
            senior, tenure, monthly, total,
            1 if gender == "Female" else 0,
            1 if gender == "Male"   else 0,
            1 if partner == "No"  else 0, 1 if partner == "Yes" else 0,
            1 if dep == "No"      else 0, 1 if dep == "Yes"     else 0,
            1 if phone == "No"    else 0, 1 if phone == "Yes"   else 0,
            1 if multiline == "No" else 0,
            1 if multiline == "No phone service" else 0,
            1 if multiline == "Yes" else 0,
            1 if internet == "DSL"         else 0,
            1 if internet == "Fiber optic" else 0,
            1 if internet == "No"          else 0,
            1 if sec == "No" else 0, 1 if sec == "No internet service" else 0, 1 if sec == "Yes" else 0,
            1 if bak == "No" else 0, 1 if bak == "No internet service" else 0, 1 if bak == "Yes" else 0,
            1 if dev == "No" else 0, 1 if dev == "No internet service" else 0, 1 if dev == "Yes" else 0,
            1 if tec == "No" else 0, 1 if tec == "No internet service" else 0, 1 if tec == "Yes" else 0,
            1 if tv  == "No" else 0, 1 if tv  == "No internet service" else 0, 1 if tv  == "Yes" else 0,
            1 if mov == "No" else 0, 1 if mov == "No internet service" else 0, 1 if mov == "Yes" else 0,
            1 if contract == "Month-to-month" else 0,
            1 if contract == "One year"       else 0,
            1 if contract == "Two year"       else 0,
            1 if paperless == "No"  else 0, 1 if paperless == "Yes" else 0,
            1 if payment == "Bank transfer (automatic)" else 0,
            1 if payment == "Credit card (automatic)"   else 0,
            1 if payment == "Electronic check"          else 0,
            1 if payment == "Mailed check"              else 0,
        ]])

        pred  = model.predict(X)[0]
        prob  = model.predict_proba(X)[0]
        churn_pct = round(prob[1] * 100, 1)
        safe_pct  = round(prob[0] * 100, 1)

        if pred == 1:
            css, pct, lbl = "result-churn", f"{churn_pct}%", "Churn risk"
            desc  = "High-risk signals detected.<br>Proactive retention outreach recommended."
            bar_c = "#E24B4A"
        else:
            css, pct, lbl = "result-safe", f"{safe_pct}%", "Likely to stay"
            desc  = "Strong retention signals present.<br>Standard engagement should be sufficient."
            bar_c = "#639922"

        st.markdown(f"""
        <div class="result-wrap {css}">
            <div class="result-pct">{pct}</div>
            <div class="result-label">{lbl}</div>
            <div class="result-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

        # Risk factors
        factors = []
        if contract  == "Month-to-month":   factors.append(("Month-to-month contract",  0.28))
        if payment   == "Electronic check": factors.append(("Electronic check payment", 0.14))
        if tenure    <  12:                 factors.append(("Short tenure (<12 mo)",    0.15))
        if internet  == "Fiber optic":      factors.append(("Fiber optic internet",     0.12))
        if monthly   >  80:                 factors.append(("High monthly charges",     0.08))
        if paperless == "Yes":              factors.append(("Paperless billing",         0.05))
        if sec == "No":                     factors.append(("No online security",        0.04))
        if tec == "No":                     factors.append(("No tech support",           0.04))
        if senior == 1:                     factors.append(("Senior citizen",            0.04))
        factors = sorted(factors, key=lambda x: -x[1])[:5]

        if factors:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Key risk factors</div>', unsafe_allow_html=True)
            max_w = factors[0][1]
            for label, w in factors:
                bar_pct = round((w / max_w) * 100)
                st.markdown(f"""
                <div class="factor">
                    <div class="factor-top"><span>{label}</span><span>{round(w*100)}%</span></div>
                    <div class="factor-bg"><div class="factor-fill" style="width:{bar_pct}%;background:{bar_c}"></div></div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Summary metrics
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Churn probability",     f"{churn_pct}%")
        m2.metric("Retention probability", f"{safe_pct}%")
        m3, m4 = st.columns(2)
        m3.metric("Tenure",           f"{tenure} mo")
        m4.metric("Monthly charges",  f"${monthly:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-wrap result-neutral">
            <div class="result-pct">—</div>
            <div class="result-label">Awaiting input</div>
            <div class="result-desc">Fill in the customer details on the left<br>and click <b>Predict churn</b>.</div>
        </div>""", unsafe_allow_html=True)