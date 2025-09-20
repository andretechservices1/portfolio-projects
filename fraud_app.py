import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import random
import plotly.express as px

# ---------------------------------------------------
# Load model + scaler
# ---------------------------------------------------
HERE = os.path.dirname(__file__)
model = joblib.load(os.path.join(HERE, "fraud_model.joblib"))
scaler = joblib.load(os.path.join(HERE, "fraud_scaler.joblib"))

# ---------------------------------------------------
# Load dataset (for visualizations only)
# ---------------------------------------------------
if os.path.exists(os.path.join(HERE, "creditcard_sample.csv")):
    data_path = os.path.join(HERE, "creditcard_sample.csv")
elif os.path.exists(os.path.join(HERE, "creditcard.csv")):
    data_path = os.path.join(HERE, "creditcard.csv")
else:
    st.error("❌ No dataset found. Please add creditcard_sample.csv or creditcard.csv")
    st.stop()

df = pd.read_csv(data_path)

# Toggle: full dataset vs 10k sample (only applies if full dataset is present)
if "creditcard.csv" in data_path:
    use_sample = st.sidebar.checkbox("Use 10,000 sample for speed", value=True)
    if use_sample and len(df) > 10000:
        df = df.sample(n=10000, random_state=42)

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("💳 Fraud Detection App")

# -------------------------------
# Single Transaction Prediction
# -------------------------------
st.header("🔹 Single Transaction Input")

inputs = []
for col in df.drop("Class", axis=1).columns:
    val = st.number_input(f"{col}", value=0.0)
    inputs.append(val)

inputs = np.array(inputs).reshape(1, -1)
inputs_scaled = scaler.transform(inputs)

if st.button("🔍 Predict Fraud (Single)"):
    prob = model.predict_proba(inputs_scaled)[0][1]
    pred = model.predict(inputs_scaled)[0]

    st.subheader("Prediction Result")
    st.write(f"**Fraud Probability:** {prob:.2%}")

    if pred == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

# -------------------------------
# Batch Transaction Simulator
# -------------------------------
st.header("📦 Batch Transaction Simulator")

feature_names = df.drop("Class", axis=1).columns

def generate_random_transaction():
    values = {}
    for col in feature_names:
        if col == "Time":
            values[col] = random.randint(0, 172792)
        elif col == "Amount":
            values[col] = round(random.uniform(0, 25000), 2)
        else:
            values[col] = round(random.uniform(-5, 5), 2)
    return values

batch_size = st.number_input("Number of random transactions", min_value=5, max_value=50, value=10, step=1)

if st.button("🚀 Generate & Predict Batch"):
    batch_data = [generate_random_transaction() for _ in range(batch_size)]
    batch_df = pd.DataFrame(batch_data)

    # Scale & predict
    scaled = scaler.transform(batch_df)
    batch_df["Fraud_Probability"] = model.predict_proba(scaled)[:, 1]
    batch_df["Prediction"] = model.predict(scaled)

    st.subheader("Batch Results (first 20 shown)")
    st.dataframe(batch_df.head(20))

    fraud_count = batch_df["Prediction"].sum()
    st.write(f"**Detected {fraud_count} fraudulent transactions out of {batch_size}.**")

    # Download CSV
    csv = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Batch Results as CSV",
        data=csv,
        file_name="fraud_batch_results.csv",
        mime="text/csv",
    )

    # Chart: fraud probabilities
    prob_fig = px.histogram(batch_df, x="Fraud_Probability", nbins=20,
                            title="Fraud Probability Distribution (Batch)")
    st.plotly_chart(prob_fig)

# -------------------------------
# Dataset Visualizations
# -------------------------------
st.header("📊 Dataset Visualizations")

# Fraud vs Legit count
st.subheader("Fraud vs Legit Count (from dataset)")
if "Class" in df.columns:
    count_data = df["Class"].value_counts().reset_index()
    count_data.columns = ["Prediction", "count"]
    count_data["Prediction"] = count_data["Prediction"].map({0: "Legit", 1: "Fraud"})

    bar_fig = px.bar(
        count_data,
        x="Prediction",
        y="count",
        color="Prediction",
        text="count",
        title="Fraud vs Legit Count"
    )
    st.plotly_chart(bar_fig)

# Fraud probability distribution (dataset)
if "Class" in df.columns:
    st.subheader("Fraud Probability Distribution (from dataset)")
    X_scaled = scaler.transform(df.drop("Class", axis=1))
    y_probs = model.predict_proba(X_scaled)[:, 1]

    hist_fig = px.histogram(
        x=y_probs,
        nbins=30,
        title="Distribution of Fraud Probabilities (Dataset)",
        labels={"x": "Fraud Probability"},
        color=df["Class"].map({0: "Legit", 1: "Fraud"})
    )
    st.plotly_chart(hist_fig)
