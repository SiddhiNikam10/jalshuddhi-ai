import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. Startup Phase: Train the AI ---
JAVA_API_URL = "http://localhost:8080/api/water/all"
print(f"Fetching data from {JAVA_API_URL} to train AI...")

response = requests.get(JAVA_API_URL)
df = pd.DataFrame(response.json())

# Clean data
df = df.drop('id', axis=1)
df = df.fillna(df.mean())

# Define inputs and target
X = df.drop('potability', axis=1)
y = df['potability']
feature_names = X.columns.tolist()

# Train model
print("Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("✅ AI Model Trained Successfully!")

# --- 2. Web Server Phase ---
app = Flask(__name__)
CORS(app)  # Allows browser requests from your dashboard

@app.route('/predict', methods=['POST'])
def predict():
    try:
        incoming_data = request.json
        input_df = pd.DataFrame([incoming_data], columns=feature_names)
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "is_safe": bool(prediction == 1),
            "safety_probability": round(float(probability) * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("🚀 Starting AI Microservice on Port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)