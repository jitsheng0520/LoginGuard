#  Login Guard: AI-Powered Anomaly detection

**Login Guard** is an Intelligent Security System designed to detect fraudulent login attempts using Machine Learning. It analyzes user behavior patterns—such as geographical distance, session timing, and hardware consistency—to flag suspicious activity in real-time.

---

3 types of model is used for comparison
1. Isolation Forest
2. 


##  Key Features
* **Machine Learning Engine:** Utilizes the **Isolation Forest** algorithm for unsupervised anomaly detection.
* **Behavioral Analysis:** Evaluates 6 critical security features:
    * Geographical distance from last login.
    * Session duration and login hour.
    * Number of failed attempts.
    * IP Address and Device consistency.
* **Interactive Dashboard:** Built with **Streamlit** to allow security administrators to simulate and test login threats.

---

Technical Stack
* **Language:** Python 3.12
* **ML Libraries:** Scikit-Learn (Isolation Forest, One-Class SVM, Local Outlier Factor)
* **Data Science:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit, Joblib

---

Project Structure
* `app.py`: The main Streamlit dashboard application.
* `export_model.py`: Training script that exports the trained model and scaler as `.pkl` files.
* `model_comparison.py`: Comparative analysis of different anomaly detection algorithms.
* `login_data_with_anomalies.csv`: Synthetic dataset used for training and testing.
* `data_scaler.pkl`: The saved StandardScaler for data normalization.
* `isolation_forest_model.pkl`: The saved "brain" of the AI.

---

Run this project (CMD)
1. **Prepare the Model:**
   python export_model.py

2. **Run the Streamlit Application:**
    python -m streamlit run app.py