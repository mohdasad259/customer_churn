import os
import pickle
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Churn Oracle", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #0A0814;
    background-image:
        radial-gradient(ellipse 80% 60% at 10% 0%, rgba(120, 40, 200, 0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%, rgba(0, 210, 190, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(255, 60, 130, 0.06) 0%, transparent 70%);
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1180px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* ── HERO HEADER ── */
.hero-wrap {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 32px;
}
.hero-icon {
    width: 52px;
    height: 52px;
    border-radius: 16px;
    background: linear-gradient(135deg, #7B2FF7 0%, #E040FB 50%, #00D2BE 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
    box-shadow: 0 0 32px rgba(123, 47, 247, 0.5), 0 0 64px rgba(224, 64, 251, 0.2);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #E040FB 0%, #7B2FF7 40%, #00D2BE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1;
}
.hero-sub {
    font-size: 13px;
    color: rgba(255,255,255,0.38);
    font-family: 'DM Mono', monospace;
    font-weight: 300;
    letter-spacing: 0.5px;
    margin-top: 5px;
}

/* ── CARDS ── */
.card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 22px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.3);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}

/* ── RESULT PANEL ── */
.result-wrap {
    border-radius: 20px;
    padding: 32px 24px 28px;
    text-align: center;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.result-churn {
    background: linear-gradient(145deg, rgba(163, 45, 45, 0.25) 0%, rgba(226, 75, 74, 0.1) 100%);
    border: 1px solid rgba(226, 75, 74, 0.4);
}
.result-churn::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(226, 75, 74, 0.08) 0%, transparent 60%);
    pointer-events: none;
}
.result-safe {
    background: linear-gradient(145deg, rgba(0, 210, 190, 0.15) 0%, rgba(59, 109, 17, 0.1) 100%);
    border: 1px solid rgba(0, 210, 190, 0.35);
}
.result-safe::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(0, 210, 190, 0.07) 0%, transparent 60%);
    pointer-events: none;
}
.result-neutral {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
}

.result-pct {
    font-family: 'Syne', sans-serif;
    font-size: 76px;
    font-weight: 800;
    letter-spacing: -5px;
    line-height: 1;
    position: relative;
}
.result-churn .result-pct { color: #FF6B6B; text-shadow: 0 0 40px rgba(255, 107, 107, 0.4); }
.result-safe  .result-pct { color: #00D2BE; text-shadow: 0 0 40px rgba(0, 210, 190, 0.4); }
.result-neutral .result-pct { color: rgba(255,255,255,0.15); }

.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 500;
    margin-top: 10px;
}
.result-churn   .result-label { color: rgba(255, 107, 107, 0.8); }
.result-safe    .result-label { color: rgba(0, 210, 190, 0.8); }
.result-neutral .result-label { color: rgba(255,255,255,0.25); }

.result-desc {
    font-size: 13px;
    color: rgba(255,255,255,0.45);
    margin-top: 14px;
    line-height: 1.7;
}

/* ── RISK FACTORS ── */
.factor { margin-bottom: 13px; }
.factor-top {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: rgba(255,255,255,0.55);
    margin-bottom: 6px;
    font-family: 'Outfit', sans-serif;
}
.factor-top span:last-child {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: rgba(255,255,255,0.35);
}
.factor-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    height: 4px;
}
.factor-fill { height: 100%; border-radius: 99px; }

/* ── FORM CONTROLS ── */
[data-baseweb="select"] > div {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.05) !important;
    font-size: 13px !important;
    color: rgba(255,255,255,0.85) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-baseweb="select"] > div:hover {
    border-color: rgba(123, 47, 247, 0.6) !important;
    background: rgba(123, 47, 247, 0.08) !important;
}
[data-baseweb="select"] [aria-selected="true"] {
    background: rgba(123, 47, 247, 0.2) !important;
}

input[type="number"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.05) !important;
    color: rgba(255,255,255,0.85) !important;
    font-size: 13px !important;
}
input[type="number"]:focus {
    border-color: rgba(123, 47, 247, 0.7) !important;
    box-shadow: 0 0 0 3px rgba(123, 47, 247, 0.15) !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(135deg, #7B2FF7, #E040FB) !important;
    border-color: #E040FB !important;
    box-shadow: 0 0 12px rgba(224, 64, 251, 0.5) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    color: #E040FB !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}
/* Slider track fill */
[data-testid="stSlider"] [role="progressbar"] {
    background: linear-gradient(90deg, #7B2FF7, #E040FB) !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #7B2FF7 0%, #E040FB 55%, #00D2BE 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 28px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    font-family: 'Syne', sans-serif !important;
    letter-spacing: 0.5px !important;
    width: 100%;
    box-shadow: 0 4px 24px rgba(123, 47, 247, 0.45), 0 0 48px rgba(224, 64, 251, 0.2) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(123, 47, 247, 0.55), 0 0 64px rgba(224, 64, 251, 0.25) !important;
}
.stButton > button:active { transform: translateY(0px); }

/* ── LABELS & TEXT ── */
label, [data-testid="stWidgetLabel"] p {
    font-size: 12px !important;
    color: rgba(255,255,255,0.4) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    letter-spacing: 0.2px !important;
}

/* ── METRICS ── */
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: rgba(255,255,255,0.9) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    color: rgba(255,255,255,0.3) !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* ── MISC ── */
.stAlert { border-radius: 12px !important; }

div[data-testid="column"] > div:first-child { width: 100%; }

p { color: rgba(255,255,255,0.7); }
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

# ── Hero Header ────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-icon">◈</div>
    <div>
        <div class="hero-title">Churn Oracle</div>
        <div class="hero-sub">// AI-powered customer retention intelligence</div>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

# ══════════════════════════════════════════════════════
# LEFT — Form
# ══════════════════════════════════════════════════════
with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">◎ Customer identity</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: gender  = st.selectbox("Gender",         ["Male", "Female"])
    with c2: senior  = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with c3: partner = st.selectbox("Partner",        ["Yes", "No"])
    c4, c5 = st.columns(2)
    with c4: dep    = st.selectbox("Dependents", ["Yes", "No"])
    with c5: tenure = st.slider("Tenure (months)", 0, 72, 12)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">◎ Phone services</div>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    with c6: phone     = st.selectbox("Phone service",  ["Yes", "No"])
    with c7: multiline = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">◎ Internet & add-ons</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-label">◎ Billing & payments</div>', unsafe_allow_html=True)
    c14, c15 = st.columns(2)
    with c14: contract  = st.selectbox("Contract type",       ["Month-to-month", "One year", "Two year"])
    with c15: paperless = st.selectbox("Paperless billing",   ["Yes", "No"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    c16, c17 = st.columns(2)
    with c16: monthly = st.number_input("Monthly charges ($)",  0.0, 200.0,   65.0, step=0.01)
    with c17: total   = st.number_input("Total charges ($)",    0.0, 10000.0, 1200.0, step=0.01)
    st.markdown('</div>', unsafe_allow_html=True)

    predict = st.button("⚡  Run Churn Analysis")

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
            css, pct, lbl = "result-churn", f"{churn_pct}%", "Churn risk detected"
            desc  = "High-risk signals identified.<br>Immediate retention outreach is recommended."
            bar_grad = "linear-gradient(90deg, #A32D2D, #FF6B6B)"
        else:
            css, pct, lbl = "result-safe", f"{safe_pct}%", "Customer likely to stay"
            desc  = "Strong loyalty signals present.<br>Standard engagement should be sufficient."
            bar_grad = "linear-gradient(90deg, #0F6E56, #00D2BE)"

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
        if tenure    <  12:                 factors.append(("Short tenure < 12 months", 0.15))
        if internet  == "Fiber optic":      factors.append(("Fiber optic internet",     0.12))
        if monthly   >  80:                 factors.append(("High monthly charges",     0.08))
        if paperless == "Yes":              factors.append(("Paperless billing",         0.05))
        if sec == "No":                     factors.append(("No online security",        0.04))
        if tec == "No":                     factors.append(("No tech support",           0.04))
        if senior == 1:                     factors.append(("Senior citizen",            0.04))
        factors = sorted(factors, key=lambda x: -x[1])[:5]

        if factors:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">◎ Key risk signals</div>', unsafe_allow_html=True)
            max_w = factors[0][1]
            for label, w in factors:
                bar_pct = round((w / max_w) * 100)
                st.markdown(f"""
                <div class="factor">
                    <div class="factor-top">
                        <span>{label}</span>
                        <span>{round(w * 100)}%</span>
                    </div>
                    <div class="factor-bg">
                        <div class="factor-fill" style="width:{bar_pct}%;background:{bar_grad}"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Summary metrics
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">◎ Prediction summary</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Churn probability",     f"{churn_pct}%")
        m2.metric("Retention probability", f"{safe_pct}%")
        m3, m4 = st.columns(2)
        m3.metric("Tenure",          f"{tenure} mo")
        m4.metric("Monthly charges", f"${monthly:.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-wrap result-neutral">
            <div class="result-pct">◈</div>
            <div class="result-label">Awaiting analysis</div>
            <div class="result-desc">Configure customer profile on the left<br>and run the churn analysis.</div>
        </div>""", unsafe_allow_html=True)