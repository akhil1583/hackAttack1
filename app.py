from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS so React/other frontend can access

# Load the trained model
model = joblib.load('fraud_detection_model.pkl')

@app.route("/")
def home():
    return render_template("index.html")  # index.html must be inside 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
