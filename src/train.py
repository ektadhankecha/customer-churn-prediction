import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

PROCESSED_DATA_PATH = "data/processed"
MODEL_PATH = "models"

def train_model():
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Load processed data
    X_train = pd.read_csv(f"{PROCESSED_DATA_PATH}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DATA_PATH}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DATA_PATH}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{PROCESSED_DATA_PATH}/y_test.csv").values.ravel()

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("✅ Model trained successfully")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_file = os.path.join(MODEL_PATH, "churn_model.pkl")
    joblib.dump(model, model_file)

    print(f"💾 Model saved at: {model_file}")

if __name__ == "__main__":
    train_model()
