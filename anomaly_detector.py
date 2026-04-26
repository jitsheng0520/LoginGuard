import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Load the dataset
# We use the dataset containing both normal and anomalous data to test it
df = pd.read_csv('login_data_with_anomalies.csv')

# 2. Separate Features and Labels
X = df.drop('label', axis=1)
y_true = df['label'].map({'normal': 0, 'anomaly': 1}) # Map text to 0 (normal) and 1 (anomaly)

# 3. Preprocess and Scale the Data
# Machine learning models work best when continuous numbers are scaled
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour']] = \
    scaler.fit_transform(X[['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour']])

# 4. Train the Isolation Forest Model
# We set contamination to the rough percentage of anomalies we expect in the system
contamination_rate = len(df[df['label'] == 'anomaly']) / len(df)
clf = IsolationForest(contamination=contamination_rate, random_state=42)

# Fit the model (Notice we don't pass y_true! It finds anomalies unsupervised)
clf.fit(X_scaled)

# 5. Make Predictions
# IsolationForest outputs -1 for anomalies and 1 for normal data. We convert this to 1 and 0.
preds = clf.predict(X_scaled)
y_pred = np.where(preds == -1, 1, 0)

# 6. Evaluate the Model
print("--- Login Guard: Model Evaluation ---")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

# 7. Visualize the Anomalies using PCA (Dimensionality Reduction)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
# Using colors that match your portfolio theme!
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['label'], 
                palette={'normal': '#667eea', 'anomaly': '#e05f3a'}, alpha=0.7)
plt.title('Login Guard: Anomaly Detection in User Authentication')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Authentication Status')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('login_guard_visualization.png')
print("Visualization saved as 'login_guard_visualization.png'")
plt.show()