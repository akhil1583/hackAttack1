from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import pandas as pd
from fraud_model import train_model_from_csv, predict_csv

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Temporary file path for saving prediction results
TEMP_TEST_CSV = "temp_test_result.csv"

# ====== Routes ======

# Home route to serve frontend
@app.route("/")
def index():
    return render_template("index.html")

# Route to train the model from uploaded training data
@app.route("/train", methods=["POST"])
def train():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    try:
        train_model_from_csv(file)
        return jsonify({"message": "Model trained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to make predictions from uploaded test data
@app.route("/predict_csv", methods=["POST"])
def predict_csv_route():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    try:
        result_df = predict_csv(file)
        result_df.to_csv(TEMP_TEST_CSV, index=False)
        return send_file(TEMP_TEST_CSV, as_attachment=True, download_name="fraud_predictions.csv")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
