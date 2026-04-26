import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load your data
df = pd.read_csv('login_data_with_anomalies.csv')

# 2. DEFINE THE MASTER ORDER (Strict Alignment)
feature_cols = ['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour', 'ip_changed', 'new_device']

# 3. Separate Features using the Master Order
X = df[feature_cols] 

# 4. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train the Isolation Forest
contamination_rate = len(df[df['label'] == 'anomaly']) / len(df)
model = IsolationForest(contamination=contamination_rate, random_state=42)
model.fit(X_scaled)

# 6. EXPORT
joblib.dump(model, 'isolation_forest_model.pkl')
joblib.dump(scaler, 'data_scaler.pkl')

print("✅ SUCCESS: Master alignment complete. Files saved.")