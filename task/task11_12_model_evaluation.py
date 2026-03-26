import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from task._common import DEFAULT_CSV, load_df


def main():
    df = load_df(DEFAULT_CSV)

    X = df[["Usage Frequency", "Payment Delay"]].astype(float)
    y_reg = df["Total Spend"].astype(float)
    y_cls = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    print("\nRegression Evaluation (Total Spend prediction)")
    print("MSE:", round(float(mean_squared_error(y_test, pred)), 3))
    print("MAE:", round(float(mean_absolute_error(y_test, pred)), 3))
    print("R2 :", round(float(r2_score(y_test, pred)), 3))

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cls, test_size=0.3, random_state=42, stratify=y_cls
    )
    cls = LogisticRegression(max_iter=1000)
    cls.fit(X_train_c, y_train_c)
    pred_c = cls.predict(X_test_c)

    print("\nClassification Evaluation (Churn prediction)")
    print("Accuracy:", round(float(accuracy_score(y_test_c, pred_c)), 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test_c, pred_c))

    print("\nComparison:")
    print("- Regression metrics measure numeric prediction error.")
    print("- Classification metrics measure correct/incorrect class decisions.")


if __name__ == "__main__":
    main()

