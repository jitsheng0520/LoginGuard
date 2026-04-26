import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# Import the three different models
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# --- 1. Load and Prepare Data ---
df = pd.read_csv('login_data_with_anomalies.csv')

X = df.drop('label', axis=1)
y_true = df['label'].map({'normal': 0, 'anomaly': 1})
contamination_rate = len(df[df['label'] == 'anomaly']) / len(df)

# Scale the data
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour']] = \
    scaler.fit_transform(X[['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour']])

# Reduce dimensions for visualization (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- 2. Initialize the Models ---
models = {
    "Isolation Forest": IsolationForest(contamination=contamination_rate, random_state=42),
    "One-Class SVM": OneClassSVM(nu=contamination_rate, kernel="rbf", gamma="scale"),
    "Local Outlier Factor": LocalOutlierFactor(contamination=contamination_rate, novelty=False)
}

# --- 3. Train, Evaluate, and Plot ---
# Set up a figure with 3 side-by-side plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Algorithm Comparison: Login Guard Anomaly Detection', fontsize=16)

# Loop through our models dictionary
for i, (model_name, model) in enumerate(models.items()):
    
    # Fit the model and get predictions
    if model_name == "Local Outlier Factor":
        # LOF handles fitting and predicting in one step when novelty=False
        preds = model.fit_predict(X_scaled)
    else:
        model.fit(X_scaled)
        preds = model.predict(X_scaled)
    
    # Convert predictions (-1 for anomaly, 1 for normal) to 1 and 0 to match y_true
    y_pred = np.where(preds == -1, 1, 0)
    
    # Print the evaluation metrics to the terminal
    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"{'='*40}")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Plot the results on the corresponding subplot
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], 
        hue=y_pred, 
        palette={0: '#667eea', 1: '#e05f3a'}, # Blue for normal, Orange/Red for anomaly
        alpha=0.7, 
        ax=axes[i]
    )
    axes[i].set_title(model_name)
    axes[i].set_xlabel('Principal Component 1')
    axes[i].set_ylabel('Principal Component 2')
    
    # Update legend labels to be readable
    handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend(handles=handles, labels=['Normal', 'Anomaly'], title='Prediction')

plt.tight_layout()
plt.savefig('model_comparison_results.png')
print("\nVisualization saved as 'model_comparison_results.png'")
plt.show()