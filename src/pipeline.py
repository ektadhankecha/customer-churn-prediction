import os
import shutil

from src.preprocess import preprocess_data
from src.train import train_model


RAW_DATA_PATH = "data/raw"
SAMPLE_DATA_PATH = "data/sample/customer_churn_sample.csv"


def ensure_raw_dataset():
    """
    Ensure there is at least one CSV in data/raw.
    On Streamlit Cloud, we use a small bundled sample dataset.
    """
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    has_csv = any(f.lower().endswith(".csv") for f in os.listdir(RAW_DATA_PATH))
    if has_csv:
        return

    if not os.path.exists(SAMPLE_DATA_PATH):
        raise FileNotFoundError(
            f"Missing sample dataset at '{SAMPLE_DATA_PATH}'. "
            "Please add a CSV dataset or commit the sample file."
        )

    shutil.copyfile(SAMPLE_DATA_PATH, os.path.join(RAW_DATA_PATH, "customer_churn_sample.csv"))


def ensure_artifacts():
    """
    Create everything needed for predictions:
    - data/processed/*.csv + label_encoders.pkl + scaler.pkl
    - models/churn_model.pkl
    """
    ensure_raw_dataset()
    preprocess_data()
    train_model()

