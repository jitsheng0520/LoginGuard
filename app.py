import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- 1. Page Configuration ---
st.set_page_config(page_title="Login Guard Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ Login Guard: AI Anomaly Detection")
st.write("This interactive dashboard monitors user login behaviors and uses an Isolation Forest machine learning model to detect suspicious activity in real-time.")

# --- 2. Load Data ---
# st.cache_data prevents the app from reloading the CSV every time you click a button
@st.cache_data
def load_data():
    return pd.read_csv('login_data_with_anomalies.csv')

df = load_data()

# --- 3. Sidebar Controls ---
st.sidebar.header("Model Settings")
st.sidebar.write("Adjust the sensitivity of the AI.")
# This slider lets you interactively change the contamination rate!
contamination_rate = st.sidebar.slider("Expected Anomaly Rate", min_value=0.01, max_value=0.10, value=0.048, step=0.005)

# --- 4. Prepare and Train Model ---
X = df.drop('label', axis=1) # Drop the true labels to let the AI find anomalies unsupervised

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour']] = \
    scaler.fit_transform(X[['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour']])

# Train model based on the slider value
clf = IsolationForest(contamination=contamination_rate, random_state=42)
clf.fit(X_scaled)

# Get predictions (-1 is anomaly, 1 is normal)
preds = clf.predict(X_scaled)
df['Predicted_Anomaly'] = np.where(preds == -1, 'Yes', 'No')

# --- 5. Dashboard Metrics (Top Row) ---
col1, col2, col3 = st.columns(3)
total_logins = len(df)
total_anomalies = len(df[df['Predicted_Anomaly'] == 'Yes'])

col1.metric(label="Total Logins Analyzed", value=total_logins)
col2.metric(label="Threats Detected", value=total_anomalies)
# Change status color/text based on threat volume
if total_anomalies > 60:
    col3.metric(label="System Status", value="High Alert 🚨")
else:
    col3.metric(label="System Status", value="Secure ✅")

st.divider()

# --- 6. Visualizations & Data Table ---
col_graph, col_table = st.columns([1.5, 1])

with col_graph:
    st.subheader("Login Behavior Cluster Analysis")
    st.write("Red dots indicate behaviors that deviate significantly from standard user patterns.")
    
    # PCA for 2D plotting
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Predicted_Anomaly'], 
                    palette={'No': '#667eea', 'Yes': '#e05f3a'}, alpha=0.7, ax=ax)
    ax.set_xlabel('Behavior Vector 1')
    ax.set_ylabel('Behavior Vector 2')
    ax.legend(title='Is Anomaly?')
    
    # Display the plot in Streamlit
    st.pyplot(fig)

with col_table:
    st.subheader("Actionable Threat Log")
    st.write("Details of the flagged login attempts.")
    
    # Filter the dataframe to only show the predicted anomalies
    anomalies_df = df[df['Predicted_Anomaly'] == 'Yes'].drop('label', axis=1)
    
    # Display as an interactive dataframe
    st.dataframe(anomalies_df, use_container_width=True, height=350)