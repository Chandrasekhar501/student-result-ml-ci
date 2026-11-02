from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path("student_result_model.pkl")
FEATURES = ["attendance", "internal_marks", "assignment_marks", "previous_score"]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run train_model.py before starting the Flask API.")
    return joblib.load(MODEL_PATH)

@app.get("/")
@app.get("/health")
def health_check():
    return jsonify({"status": "ok", "service": "student-result-prediction"})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "JSON request body is required"}), 400
    missing = [name for name in FEATURES if name not in data]
    if missing:
        return jsonify({"error": "Missing required fields", "missing_fields": missing}), 400
    try:
        sample = pd.DataFrame([{name: float(data[name]) for name in FEATURES}])
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numeric"}), 400
    code = int(load_model().predict(sample)[0])
    return jsonify({"prediction": "PASS" if code == 1 else "FAIL", "prediction_code": code})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
