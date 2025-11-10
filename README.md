🫀 Cardiovascular Disease Prediction Web App

A machine-learning-powered web application that predicts the risk of cardiovascular disease (CVD) based on key health indicators such as age, BMI, blood pressure, cholesterol, glucose level, smoking habits, and physical activity.

Built using Python (Flask) for the backend and Tailwind CSS for a modern, responsive frontend design.

🚀 Project Overview

This project simulates a real-world healthcare predictive model that analyzes clinical and lifestyle data to estimate a patient’s likelihood of developing cardiovascular disease.
The model uses Logistic Regression and Random Forest algorithms, trained on synthetic yet realistic health data generated with domain-based rules.

The frontend allows users to input their details and instantly get a risk prediction, probability score, and risk factor breakdown.

🧩 Features

🔍 Predict CVD risk instantly using a trained ML model.

🧠 Dual Model Support – Logistic Regression and Random Forest (optional).

📊 Automatic Risk Factor Detection – identifies elevated health indicators.

🧮 Probability-based Results – shows low, medium, or high risk.

💅 Modern Responsive UI – built with Tailwind CSS for a clean look.

⚙️ Modular & Scalable Codebase – easy to extend with new models or features.


🛠️ Tools & Libraries Used
Category                  Tools / Libraries                         	Description
Programming Language        	Python              	        Core language for data generation, model training, and backend
Web Framework	                Flask	                        Lightweight web framework for routing and serving predictions
Data Handling       	    pandas, NumPy                   	Used for data manipulation, feature generation, and numerical computations
Machine Learning        	scikit-learn	                    Provides Logistic Regression, Random Forest, and scaling utilities
Model Persistence	          joblib                         	Saves and loads trained ML models efficiently
Frontend Styling         	Tailwind CSS                    	Provides responsive, modern UI components

💡 How It Works

1. Synthetic Data Generation

   Randomized yet realistic values for health parameters (Age, BMI, BP, etc.) are generated.

   A weighted risk score simulates true CVD probability using a logistic function.
   

2. Model Training

   Data is split, scaled, and trained on Logistic Regression (optionally Random Forest).

   Trained models and scalers are saved using joblib.

3. Prediction Workflow

   User submits health parameters from the web form.

   Input data is scaled using the same scaler as during training.

   Model outputs a probability and predicted class (0 = No CVD, 1 = CVD).

   Risk factors are analyzed, and a risk level badge (Low, Medium, High) is displayed



   | Parameter             | Normal Range | High Risk Trigger |
| --------------------- | ------------ | ----------------- |
| **BMI**               | 18.5 – 24.9  | > 24.9            |
| **Blood Pressure**    | ≤ 120 mmHg   | > 120 mmHg        |
| **Cholesterol**       | ≤ 200 mg/dL  | > 200 mg/dL       |
| **Glucose**           | ≤ 140 mg/dL  | > 140 mg/dL       |
| **Smoking**           | No           | Yes               |
| **Physical Activity** | > 0 hrs/day  | 0 hrs/day         |

If multiple triggers are detected, the app highlights “High Risk” and displays a list of concerning factors.

EXAMPLE INPUT 

| Age | BMI  | Blood Pressure | Cholesterol | Glucose | Smoking | Physical Activity |
| --- | ---- | -------------- | ----------- | ------- | ------- | ----------------- |
| 54  | 28.7 | 142            | 230         | 175     | Yes     | 0                 |


Output:

🧮 Predicted CVD Probability: Likely above 80% (based on high-risk profile)

⚠️ Risk Level: High


🧑‍💻 Author

Star Lord
Data Analytics & Machine Learning Enthusiast
📧 [irfanansari7139@gmail.com]

🌐 [LinkedIn ;- https://www.linkedin.com/in/md-irfan-2b7480226/]





