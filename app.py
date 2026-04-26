import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🛡️ Login Guard: AI Anomaly Detection")

# Load our saved brain
model = joblib.load('isolation_forest_model.pkl')
scaler = joblib.load('data_scaler.pkl')

st.subheader("Simulate a Login Attempt")
dist = st.slider("Distance from last login (km)", 0, 5000, 10)
duration = st.number_input("Session Duration (sec)", value=60)
fails = st.number_input("Failed Attempts", 0, 10, 0)
hour = st.slider("Hour of Day (0-23)", 0, 23, 12)

if st.button("Check for Threats"):
    # Prepare the data exactly how the model likes it
    input_data = pd.DataFrame([[dist, duration, fails, hour]], 
                              columns=['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour'])
    
    # Scale and Predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    
    if prediction[0] == -1:
        st.error("⚠️ ANOMALY DETECTED: This login looks suspicious!")
    else:
        st.success("✅ Access Granted: Normal behavior patterns.")