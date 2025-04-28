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
    return "UP and Running! Use /predict endpoint to make predictions."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    # Ensure valid input
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key in request"}), 400
    
    try:
        features = np.array(data["features"]).reshape(1, -1)
        features = scaler.transform(features)  # Apply scaling
        prediction = model.predict(features).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        # handle form or data
        return "POST received!"
    return "Submit page"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

#docker build -t toyato
#docker run -p 5003:5000 toyato  