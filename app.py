
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Smart Water Tank",
    page_icon="💧",
    layout="wide"
)

@st.cache_resource
def load_models():
    return (
        tf.keras.models.load_model("lstm_model.keras"),
        tf.keras.models.load_model("dnn_model.keras"),
        joblib.load("xgboost_model.joblib"),
        joblib.load("water_level_scaler.joblib")
    )

st.title("Smart Water Tank Level Prediction")
st.caption("IoT simulation • ML prediction • Pump recommendation")

lstm, dnn, xgb, scaler = load_models()

st.sidebar.header("Simulated Sensor")
current = st.sidebar.slider("Current water level (%)", 0, 100, 55)
trend = st.sidebar.slider("Recent level trend", -2.0, 2.0, 0.5, 0.5)
model_choice = st.sidebar.selectbox(
    "Prediction model", ["LSTM", "DNN", "XGBoost"]
)

history = np.clip(
    current + trend * np.arange(-11, 1), 0, 100
).astype(np.float32)

scaled_history = scaler.transform(
    history.reshape(-1, 1)
).flatten().reshape(1, 12)

if model_choice == "LSTM":
    scaled_prediction = lstm.predict(
        scaled_history.reshape(1, 12, 1), verbose=0
    )[0][0]
elif model_choice == "DNN":
    scaled_prediction = dnn.predict(
        scaled_history, verbose=0
    )[0][0]
else:
    scaled_prediction = xgb.predict(scaled_history)[0]

predicted = float(scaler.inverse_transform(
    np.array([[scaled_prediction]])
)[0][0])
predicted = float(np.clip(predicted, 0, 100))

pump = "ON — refill recommended" if predicted < 30 else "OFF — level sufficient"

c1, c2, c3 = st.columns(3)
c1.metric("Current Level", f"{current}%")
c2.metric("Predicted Next Level", f"{predicted:.1f}%")
c3.metric("Pump Recommendation", pump)

st.subheader("Tank Level")
st.progress(current / 100)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, 13), history, marker="o", label="Recent levels")
ax.axhline(predicted, linestyle="--", label="Predicted next level")
ax.set_xlabel("Reading step")
ax.set_ylabel("Water level (%)")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
ax.legend()
st.pyplot(fig)

st.info(
    "Prototype mode: values are simulated. "
    "The pump recommendation does not control physical hardware."
)

