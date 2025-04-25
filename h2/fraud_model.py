import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

MODEL_PATH = "fraud_detection_model.pkl"

def train_model_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    if 'is_fraud' not in data.columns:
        raise ValueError("Training data must contain 'is_fraud' column.")

    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved.")
    return model

def predict_csv(csv_file):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet.")

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(csv_file)

    # Drop 'is_fraud' if exists
    if 'is_fraud' in df.columns:
        df = df.drop(columns=['is_fraud'])

    preds = model.predict(df)
    # 🛠 Reversed Mapping
    df['is_fraud'] = ['fraud' if p == 1 else 'not_fraud' for p in preds]

    return df
