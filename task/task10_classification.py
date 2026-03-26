import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from task._common import DEFAULT_CSV, load_df


def main():
    df = load_df(DEFAULT_CSV)

    median_sales = float(df["Total Spend"].median())
    df["Profitable"] = (df["Total Spend"] > median_sales).astype(int)

    X = df[["Usage Frequency", "Payment Delay", "Support Calls"]].astype(float)
    y = df["Profitable"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, pred))
    cm = confusion_matrix(y_test, pred)

    print("\nClassification task:")
    print("Profit -> Profitable / Not Profitable (based on median Total Spend)")
    print("Median Total Spend:", round(median_sales, 3))
    print("\nAccuracy:", round(acc, 3))
    print("\nConfusion Matrix:\n", cm)


if __name__ == "__main__":
    main()

