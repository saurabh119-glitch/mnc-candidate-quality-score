# Test imports first
try:
    import joblib
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    print("✅ All libraries loaded successfully!")
except Exception as e:
    print(f"❌ Error loading libraries: {e}")
    raise
import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# Load model & metrics
@st.cache_resource
def load_model():
    return joblib.load('quality_score_model.pkl')

@st.cache_data
def load_metrics():
    with open('model_metrics.json', 'r') as f:
        return json.load(f)

model = load_model()
metrics = load_metrics()

st.set_page_config(page_title="MNC Candidate Quality Score", page_icon="🔍", layout="wide")
st.title("🔍 Enterprise Candidate Quality Score Engine")
st.markdown("Predict 12-month job performance from hiring assessment data")

# Sidebar: Model Credibility
with st.sidebar:
    st.header("🏢 Model Credibility")
    st.metric("Test Accuracy (R²)", f"{metrics['R2']:.2f}")
    st.metric("Avg Error (MAE)", f"{metrics['MAE']:.2f}")
    st.caption(f"Trained on 10,000 synthetic profiles\nTested on {metrics['Test_Sample_Size']} held-out candidates")
    st.info("✅ Uses only job-relevant signals\n✅ No demographic data\n✅ Audit-ready")

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Candidate Assessment Scores")
    role = st.selectbox("Job Role Level", ["L1 - Entry", "L2 - Mid", "L3 - Senior"])
    exp = st.slider("Years of Experience", 0, 20, 3)
    edu = st.selectbox("Education Tier", 
                      [("Tier 1 (IIT/NIT/Top Private)", 1), 
                       ("Tier 2 (State Colleges)", 2), 
                       ("Tier 3 (Other)", 3)],
                      format_func=lambda x: x[0])
    
    st.markdown("### Assessment Scores (0-100)")
    coding = st.slider("Coding Test", 0, 100, 75)
    behavioral = st.slider("Behavioral Interview", 0, 100, 80)
    domain = st.slider("Domain Knowledge", 0, 100, 70)
    refs = st.slider("Reference Quality (1-5)", 1, 5, 4)

with col2:
    st.subheader("Performance Prediction")
    # Encode inputs
    role_enc = {"L1 - Entry": 0, "L2 - Mid": 1, "L3 - Senior": 2}[role]
    edu_tier = edu[1]
    
    features = [[exp, edu_tier, coding, behavioral, domain, refs, role_enc]]
    pred = model.predict(features)[0]
    
    # Visual gauge
    if pred >= 4.0:
        st.success(f"🌟 **{pred:.1f}/5.0**\n\nHigh Performer")
    elif pred >= 3.0:
        st.warning(f"📈 **{pred:.1f}/5.0**\n\nSolid Performer")
    else:
        st.error(f"⚠️ **{pred:.1f}/5.0**\n\nPerformance Risk")
    
    st.markdown(f"**Expected 12-mo Rating**\n\n*MAE: ±{metrics['MAE']}*")
    
    # Explain key drivers (simple version)
    st.markdown("### Key Drivers")
    drivers = sorted([
        ("Coding Test", coding),
        ("Behavioral", behavioral),
        ("Domain Knowledge", domain)
    ], key=lambda x: x[1], reverse=True)
    
    for i, (name, score) in enumerate(drivers[:2]):
        st.caption(f"{i+1}. {name}: {score}/100")

# Footer
st.markdown("---")

st.caption("Enterprise AI for Talent | Audit-Ready | Demo Only | [GitHub](https://github.com/yourname/mnc-quality-score)")
