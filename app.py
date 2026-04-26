import streamlit as st
import pandas as pd
import joblib

st.title("🛡️ Login Guard: AI Anomaly Detection")

# Load our saved brain
model = joblib.load('isolation_forest_model.pkl')
scaler = joblib.load('data_scaler.pkl')

st.subheader("Simulate a Login Attempt")

# Row 1: Numerical Inputs
col1, col2 = st.columns(2)
with col1:
    dist = st.slider("Distance from last login (km)", 0, 5000, 10)
    duration = st.number_input("Session Duration (sec)", value=60)
with col2:
    fails = st.number_input("Failed Attempts", 0, 10, 0)
    hour = st.slider("Hour of Day (0-23)", 0, 23, 12)

# Row 2: Categorical/Boolean Inputs (The missing features!)
# --- Define the inputs first ---
st.write("---")
# Make sure these names (the variables on the left) match exactly!
is_ip_different = st.checkbox("Did the IP Address change?") 
is_new_device = st.checkbox("Is this a new device?")

if st.button("Check for Threats"):
    ip_val = 1 if is_ip_different else 0
    device_val = 1 if is_new_device else 0
    
    # 1. Create the data with names
    data_dict = {
        'login_hour': [hour],
        'failed_attempts': [fails],
        'new_device': [device_val],
        'ip_changed': [ip_val],
        'geo_distance_km': [dist],
        'session_duration': [duration]
        
    }
    
    # 2. Convert to DataFrame
    input_data = pd.DataFrame(data_dict)
    
    # 3. FORCE THE ORDER: Change this list to match exactly 
    # what you saw in Step 1 (the output of feature_names_in_)
    correct_order = ['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour', 'ip_changed', 'new_device']
    input_data = input_data[correct_order]
    
    # 4. Now scale and predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    
    if prediction[0] == -1:
        st.error("⚠️ ANOMALY DETECTED!")
    else:
        st.success("✅ Access Granted.")