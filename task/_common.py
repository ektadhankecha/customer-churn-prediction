import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_CSV = "data/sample/customer_churn_sample.csv"
OUT_DIR = "task/output"


@dataclass
class Cols:
    sales: str | None = None
    profit: str | None = None
    quantity: str | None = None
    discount: str | None = None
    category: str | None = None


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def load_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def _pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None


def detect_cols(df: pd.DataFrame) -> Cols:
    return Cols(
        # For this dataset we treat "Total Spend" like Sales.
        sales=_pick(df, ["Sales", "Total Spend", "TotalSpend"]),
        # Profit doesn't exist; scripts derive a proxy when needed.
        profit=_pick(df, ["Profit"]),
        # Quantity doesn't exist; scripts use "Usage Frequency" or "Support Calls".
        quantity=_pick(df, ["Quantity", "Usage Frequency", "Support Calls"]),
        # Discount doesn't exist; scripts use "Payment Delay" as a proxy.
        discount=_pick(df, ["Discount", "Payment Delay"]),
        category=_pick(df, ["Category", "Subscription Type", "Contract Length"]),
    )


def require(df: pd.DataFrame, needed: dict[str, str | None]):
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + "\nAvailable columns: "
            + ", ".join(df.columns)
        )


def numeric_summary(s: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std(ddof=1)),
        "skew": float(x.skew()),
        "kurt": float(x.kurt()),
    }


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

