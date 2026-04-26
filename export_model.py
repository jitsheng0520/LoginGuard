import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load your data
df = pd.read_csv('login_data_with_anomalies.csv')
X = df.drop('label', axis=1)

# 2. Scale the data 
scaler = StandardScaler()
# Explicitly define columns to ensure the scaler remembers the names/order
feature_cols = ['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour', 'ip_changed', 'new_device']
X = X[feature_cols] # Force the order
X_scaled = scaler.fit_transform(X)

# 3. Train the Isolation Forest
# We use the contamination rate from your data
contamination_rate = len(df[df['label'] == 'anomaly']) / len(df)
model = IsolationForest(contamination=contamination_rate, random_state=42)
model.fit(X_scaled)

# 4. EXPORT: This creates the physical files app.py is looking for
joblib.dump(model, 'isolation_forest_model.pkl')
joblib.dump(scaler, 'data_scaler.pkl')

print("✅ Success: 'isolation_forest_model.pkl' and 'data_scaler.pkl' created!")