from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# =========================================================
# Load saved model and scaler
# =========================================================
try:
    import os
    model_path = os.path.join(os.path.dirname(__file__), "logistic_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
    
    log_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# =========================================================
# Home route
# =========================================================
@app.route('/')
def home():
    return render_template('index.html')

# =========================================================
# Prediction route (GET + POST to avoid 405 error)
# =========================================================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        try:
            # -----------------------------
            # Get input data from form
            # -----------------------------
            age = float(request.form['Age'])
            bmi = float(request.form['BMI'])
            bp = float(request.form['Blood_Pressure'])
            chol = float(request.form['Cholesterol'])
            glu = float(request.form['Glucose'])
            smoking = 1 if request.form['Smoking'] == "Yes" else 0
            activity = float(request.form['Physical_Activity'])

            # -----------------------------
            # Pre-check for high-risk inputs
            # -----------------------------
            high_risk_conditions = []

            if bmi > 24.9:
                high_risk_conditions.append("BMI above 24.9")
            if bp > 120:
                high_risk_conditions.append("Blood Pressure above 120 mmHg")
            if chol > 200:
                high_risk_conditions.append("Cholesterol above 200 mg/dL")
            if glu > 140:
                high_risk_conditions.append("Glucose above 140 mg/dL")
            if smoking == 1:
                high_risk_conditions.append("Smoking habit detected")
            if activity == 0:
                high_risk_conditions.append("No physical activity")

            # -----------------------------
            # If multiple risk factors → auto high risk
            # -----------------------------
            if len(high_risk_conditions) >= 3 or ((smoking == 1) and (activity == 0)):
                risk = "High"
                reason = ", ".join(high_risk_conditions)
                return render_template(
                    'index.html',
                    prediction="Likely above 80% (based on high-risk profile)",
                    risk_level=risk,
                    message=f"⚠️ Detected major risk factors: {reason}"
                )

            # -----------------------------
            # Else, do model prediction
            # -----------------------------
            X_input = np.array([[age, bmi, bp, chol, glu, smoking, activity]])
            X_scaled = scaler.transform(X_input)
            log_prob = log_model.predict_proba(X_scaled)[0][1]

            # Determine risk level from model
            if log_prob < 0.40:
                risk = "Low"
            elif log_prob < 0.70:
                risk = "Medium"
            else:
                risk = "High"

            return render_template(
                'index.html',
                prediction=f"{log_prob * 100:.2f}%",
                risk_level=risk,
                message=None
            )

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {e}", risk_level=None)

# =========================================================
# Run Flask app
# =========================================================
if __name__ == '__main__':
    app.run()

