import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


MODEL_PATH = "models/churn_model.pkl"
ENCODERS_PATH = "data/processed/label_encoders.pkl"
SCALER_PATH = "data/processed/scaler.pkl"
FEATURES_PATH = "data/processed/feature_names.csv"
SAMPLE_DATA_PATH = "data/sample/customer_churn_sample.csv"
OUTPUT_DIR = "data/graphs"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = pd.read_csv(FEATURES_PATH, header=None)[0].tolist()
    return model, encoders, scaler, feature_names


def prepare_sample_data(encoders, feature_names):
    df = pd.read_csv(SAMPLE_DATA_PATH)

    for col in list(df.columns):
        if "id" in col.lower():
            df = df.drop(columns=[col])

    X = df.drop(columns=["Churn"]).copy()
    y = df["Churn"].values

    for col, le in encoders.items():
        if col in X.columns:
            X[col] = le.transform(X[col])

    X = X[feature_names]
    return df, X, y


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def save_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()


def save_metrics_chart(y_true, y_pred):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, values, color=["#2ecc71", "#3498db", "#f39c12", "#9b59b6"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_bar.png"), dpi=150)
    plt.close()


def save_feature_importance(model, feature_names):
    if not hasattr(model, "coef_"):
        return

    coef = model.coef_.ravel()
    idx = np.argsort(np.abs(coef))[::-1]
    top_n = min(10, len(coef))

    top_features = [feature_names[i] for i in idx[:top_n]]
    top_coef = coef[idx[:top_n]]
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in top_coef]

    plt.figure(figsize=(8, 6))
    plt.barh(top_features, top_coef, color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient")
    plt.title("Top Feature Importance (Logistic Regression)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()


def save_churn_distribution(df):
    if "Churn" not in df.columns:
        return
    counts = df["Churn"].value_counts().sort_index()
    labels = ["No Churn (0)", "Churn (1)"]

    plt.figure(figsize=(6, 5))
    plt.pie(counts.values, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#3498db", "#e74c3c"])
    plt.title("Churn Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "churn_distribution.png"), dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, encoders, scaler, feature_names = load_artifacts()
    raw_df, X, y_true = prepare_sample_data(encoders, feature_names)
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    y_pred = model.predict(X_scaled_df)
    y_proba = model.predict_proba(X_scaled_df)[:, 1]

    save_confusion_matrix(y_true, y_pred)
    save_roc_curve(y_true, y_proba)
    save_metrics_chart(y_true, y_pred)
    save_feature_importance(model, feature_names)
    save_churn_distribution(raw_df)

    print("Graphs generated in data/graphs/:")
    for file_name in sorted(os.listdir(OUTPUT_DIR)):
        print(f"- {file_name}")


if __name__ == "__main__":
    main()

