import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# =========================================================
# 1️⃣ Generate Synthetic Data
# =========================================================
np.random.seed(42)
n_samples = 2000

data = {
    'Age': np.random.normal(50, 15, n_samples).clip(20, 90),
    'BMI': np.random.normal(25, 5, n_samples).clip(15, 45),
    'Blood_Pressure': np.random.normal(120, 20, n_samples).clip(80, 200),
    'Cholesterol': np.random.normal(200, 40, n_samples).clip(100, 400),
    'Glucose': np.random.normal(100, 25, n_samples).clip(60, 250),
    'Smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'Physical_Activity': np.random.normal(5, 2, n_samples).clip(0, 10)
}

df = pd.DataFrame(data)

# =========================================================
# 2️⃣ Define Baseline Risk Formula
# =========================================================
base_risk = (
    0.03 * df['Age'] +
    0.05 * df['BMI'] +
    0.04 * df['Blood_Pressure'] +
    0.03 * df['Cholesterol'] +
    0.02 * df['Glucose'] +
    0.20 * df['Smoking'] -
    0.05 * df['Physical_Activity']
)

# =========================================================
# 3️⃣ Add Conditional Risk Boosts
# =========================================================
risk_boost = np.zeros(n_samples)

# BMI high
risk_boost += np.where(df['BMI'] > 24.9, 3, 0)

# Blood Pressure high
risk_boost += np.where(df['Blood_Pressure'] > 120, 3, 0)

# Cholesterol high
risk_boost += np.where(df['Cholesterol'] > 200, 3, 0)

# Glucose high
risk_boost += np.where(df['Glucose'] > 140, 3, 0)

# Smoking and no physical activity (worst combo)
risk_boost += np.where((df['Smoking'] == 1) & (df['Physical_Activity'] == 0), 5, 0)

# Final combined risk score
final_risk_score = base_risk + risk_boost

# =========================================================
# 4️⃣ Convert to CVD Probability (logistic function)
# =========================================================
prob = 1 / (1 + np.exp(-(final_risk_score - 25) / 3))

# Assign CVD outcome based on probability
df['CVD'] = np.random.binomial(1, prob)

# =========================================================
# 5️⃣ Train-Test Split
# =========================================================
X = df.drop('CVD', axis=1)
y = df['CVD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================================================
# 6️⃣ Scale Features
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# =========================================================
# 7️⃣ Train Logistic Regression
# =========================================================
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)

# =========================================================
# 8️⃣ Save Model and Scaler
# =========================================================
joblib.dump(log_model, 'logistic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model trained and saved successfully!")
