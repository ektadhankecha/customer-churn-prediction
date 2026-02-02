import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

TARGET_COL = "Churn"

def preprocess_data():
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Load raw CSV
    csv_file = next(
        f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".csv")
    )
    df = pd.read_csv(os.path.join(RAW_DATA_PATH, csv_file))

    # Drop customer ID column if present
    for col in df.columns:
        if "id" in col.lower():
            df.drop(columns=[col], inplace=True)

    # Encode target
    if df[TARGET_COL].dtype == "object":
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # Encode categorical features
    categorical_cols = df.select_dtypes(include="object").columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    joblib.dump(label_encoders, f"{PROCESSED_DATA_PATH}/label_encoders.pkl")

    # Split
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
        f"{PROCESSED_DATA_PATH}/X_train.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
        f"{PROCESSED_DATA_PATH}/X_test.csv", index=False
    )
    y_train.to_csv(f"{PROCESSED_DATA_PATH}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DATA_PATH}/y_test.csv", index=False)

    feature_names = X.columns.tolist()
    pd.Series(feature_names).to_csv(
        f"{PROCESSED_DATA_PATH}/feature_names.csv",
        index=False,
        header=False
    )

    print("✅ Data preprocessing completed")

if __name__ == "__main__":
    preprocess_data()
