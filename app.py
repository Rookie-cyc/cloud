from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model and scaler
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Heart Disease Prediction API is up! Use the /predict endpoint."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key in request"}), 400

    features = data["features"]

    # Check that 13 features (excluding 'target') are provided
    if len(features) != 13:
        return jsonify({
            "error": "Expected 13 features (excluding 'target')",
            "received": len(features)
        }), 400

    try:
        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled).tolist()[0]
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        return jsonify({
            "prediction": int(prediction),
            "result": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == 'POST':
        return "POST received!"
    return "Submit page"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
