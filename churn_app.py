# -------------------------------
# Customer Churn Prediction App
# Works with churn_model.joblib (Pipeline: ColumnTransformer + Classifier)
# -------------------------------
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------- Page setup ----------
st.set_page_config(page_title="Customer Churn Prediction App", layout="centered")

# Subtle centering and max-width
st.markdown(
    """
    <style>
      .main .block-container {
        max-width: 1000px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
      }
      label, .stSelectbox label, .stNumberInput label {
        font-weight: 600;
      }
      .result-card {
        padding: 1rem 1.25rem;
        border-radius: 0.5rem;
        border: 1px solid #e6e6e6;
        background: #fafafa;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Customer Churn Prediction App")
st.write("Fill out the customer details below and click **Predict Churn**.")

# ---------- Load model pipeline ----------
MODEL_PATH = Path("churn_model.joblib")
if not MODEL_PATH.exists():
    st.error(
        "Model file `churn_model.joblib` was not found in the project root. "
        "Please run `train_model.py` to generate it, then refresh this app."
    )
    st.stop()

pipeline = joblib.load(MODEL_PATH)

# ---------- Feature schema (must match training exactly) ----------
FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Choice sets taken from Telco-Customer-Churn dataset values
GENDER_OPTS = ["Female", "Male"]
YES_NO = ["Yes", "No"]
YES_NO_NIS = ["No internet service", "No", "Yes"]  # order matches typical dataset variety
ML_OPTS = ["No phone service", "No", "Yes"]
INT_SVC = ["DSL", "Fiber optic", "No"]
CONTRACT = ["Month-to-month", "One year", "Two year"]
PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

# ---------- Form (two centered columns) ----------
with st.form("churn_form"):
    col1, col2 = st.columns(2, gap="large")

    with col1:
        gender = st.selectbox("Gender", GENDER_OPTS, index=0)
        senior = st.selectbox("Senior Citizen", [0, 1], index=0, help="0 = No, 1 = Yes")
        partner = st.selectbox("Partner", YES_NO, index=0)
        dependents = st.selectbox("Dependents", YES_NO, index=0)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12, step=1)
        phone = st.selectbox("Phone Service", YES_NO, index=0)
        multi = st.selectbox("Multiple Lines", ML_OPTS, index=0)
        internet = st.selectbox("Internet Service", INT_SVC, index=0)
        onsec = st.selectbox("Online Security", YES_NO_NIS, index=0)
        onbkp = st.selectbox("Online Backup", YES_NO_NIS, index=0)

    with col2:
        device = st.selectbox("Device Protection", YES_NO_NIS, index=0)
        tech = st.selectbox("Tech Support", YES_NO_NIS, index=0)
        stv = st.selectbox("Streaming TV", YES_NO_NIS, index=0)
        smov = st.selectbox("Streaming Movies", YES_NO_NIS, index=0)
        contract = st.selectbox("Contract", CONTRACT, index=0)
        paperless = st.selectbox("Paperless Billing", YES_NO, index=0)
        pay = st.selectbox("Payment Method", PAYMENT, index=0)
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=50.0, step=1.0, format="%.2f")
        total = st.number_input("Total Charges", min_value=0.0, max_value=20000.0, value=1000.0, step=10.0, format="%.2f")

    submitted = st.form_submit_button("🔮 Predict Churn")

# ---------- Build input row as DataFrame in training order ----------
def build_row() -> pd.DataFrame:
    row = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": onsec,
        "OnlineBackup": onbkp,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": stv,
        "StreamingMovies": smov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": pay,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    # Ensure correct column order
    df = pd.DataFrame([row])[FEATURES].copy()

    # Strong dtype guarantees for numeric fields
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0).astype(float)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0).astype(float)
    return df

# ---------- Predict ----------
if submitted:
    X = build_row()
    try:
        # Use predict_proba if classifier supports it; otherwise fall back to decision_function/predict
        proba = None
        if hasattr(pipeline, "predict_proba"):
            proba = float(pipeline.predict_proba(X)[0][1])
        else:
            # Some models may not implement predict_proba
            if hasattr(pipeline, "decision_function"):
                # Convert decision score to pseudo-probability via logistic transform
                score = float(pipeline.decision_function(X)[0])
                proba = 1.0 / (1.0 + np.exp(-score))
            else:
                pred = int(pipeline.predict(X)[0])
                proba = 0.75 if pred == 1 else 0.25  # reasonable fallback

        st.subheader("Prediction Result")
        with st.container():
            if proba >= 0.5:
                st.error(f"⚠️ This customer is **likely to churn**. (Probability: {proba:.2f})")
            else:
                st.success(f"✅ This customer is **likely to stay**. (Probability: {proba:.2f})")

    except Exception as e:
        # Show a clean message without leaking internal stack traces
        st.error("Prediction failed. Please verify the inputs and try again.")
        # If you want a minimal detail for your own debugging while keeping UI clean,
        # temporarily uncomment the next line:
        # st.caption(str(e))








