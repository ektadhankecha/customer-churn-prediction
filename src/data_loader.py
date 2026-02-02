import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "muhammadshahidazeem/customer-churn-dataset"

def fetch_customer_churn(save_path="data/raw"):
    os.makedirs(save_path, exist_ok=True)

    # Authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    print("⬇️ Downloading dataset from Kaggle...")
    api.dataset_download_files(
        DATASET,
        path=save_path,
        unzip=True
    )

    # Load CSV
    for file in os.listdir(save_path):
        if file.endswith(".csv"):
            csv_path = os.path.join(save_path, file)
            print(f"✅ Dataset loaded: {csv_path}")
            return pd.read_csv(csv_path)

    raise FileNotFoundError("❌ No CSV file found in dataset")

if __name__ == "__main__":
    df = fetch_customer_churn()
    print(df.head())
