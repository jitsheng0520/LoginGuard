import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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
feature_cols = ['geo_distance_km', 'session_duration', 'failed_attempts', 'login_hour', 'ip_changed', 'new_device']
X = X[feature_cols] # Ensure order
X_scaled = scaler.fit_transform(X) # Scale everything together

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
# Loop through our models dictionary
for model_name, model in models.items():
    
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
    
    # Create a new figure for THIS specific model
    plt.figure(figsize=(8, 6))
    
    # Plot the results
    ax = sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], 
        hue=y_pred, 
        palette={0: '#667eea', 1: '#e05f3a'}, # Blue for normal, Orange/Red for anomaly
        alpha=0.7
    )
    plt.title(f'Login Guard Anomaly Detection: {model_name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Update legend labels to be readable
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=['Normal', 'Anomaly'], title='Prediction')

    plt.tight_layout()
    
    # Create a dynamic filename based on the model's name (replaces spaces with underscores)
    filename = f"{model_name.replace(' ', '_').lower()}_results.png"
    plt.savefig(filename)
    print(f"Visualization successfully saved as '{filename}'")
    
    # Close the figure so the next iteration starts with a blank canvas
    plt.close()

print("\nAll models evaluated and images generated!")

# We use joblib to save the model and the scaler
joblib.dump(models["Isolation Forest"], 'isolation_forest_model.pkl')
joblib.dump(scaler, 'data_scaler.pkl')
print("Model and Scaler exported successfully!")