import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="RetainAI Monitor", layout="wide")
st.title("📈 RetainAI Monitoring Dashboard")
st.markdown("Real‑time observability for model drift, accuracy, and latency.")

# ---- Data Drift (simulated) ----
st.subheader("Data Drift Over Time")
drift_data = pd.DataFrame({
    'Week': ['W1', 'W2', 'W3', 'W4', 'W5'],
    'Drift Share': [0.05, 0.07, 0.12, 0.11, 0.15]
})
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(drift_data['Week'], drift_data['Drift Share'], color='orange')
ax.set_ylabel('Drift Share')
ax.set_title('Feature Drift Share (weekly)')
st.pyplot(fig)

# ---- Model Accuracy Trend ----
st.subheader("Model Accuracy (Test Set)")
accuracy_data = pd.DataFrame({
    'Week': ['W1', 'W2', 'W3', 'W4', 'W5'],
    'Accuracy': [0.88, 0.875, 0.87, 0.865, 0.86]
})
st.line_chart(accuracy_data.set_index('Week'))

# ---- Prediction Latency (live simulated) ----
st.subheader("Prediction Latency (ms)")
latency = np.random.normal(200, 30, 50)
st.line_chart(latency)

st.caption("Dashboard updates every 5 seconds with live data (simulated).")