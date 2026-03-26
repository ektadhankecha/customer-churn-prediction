import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from task._common import DEFAULT_CSV, load_df


def main():
    df = load_df(DEFAULT_CSV)

    X = df[["Usage Frequency", "Payment Delay"]].astype(float)
    y_reg = df["Total Spend"].astype(float)
    y_cls = df["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train[["Usage Frequency"]], y_train)
    pred_train = lr.predict(X_train[["Usage Frequency"]])
    pred_test = lr.predict(X_test[["Usage Frequency"]])
    print("\n(i) Simple Linear Regression (Total Spend ~ Usage Frequency)")
    print("Train MSE:", round(mean_squared_error(y_train, pred_train), 3))
    print("Test  MSE:", round(mean_squared_error(y_test, pred_test), 3))
    print("Test  R2 :", round(r2_score(y_test, pred_test), 3))

    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    pred_train2 = mlr.predict(X_train)
    pred_test2 = mlr.predict(X_test)
    print("\n(ii) Multi Linear Regression (Total Spend ~ Usage Frequency + Payment Delay)")
    print("Train MSE:", round(mean_squared_error(y_train, pred_train2), 3))
    print("Test  MSE:", round(mean_squared_error(y_test, pred_test2), 3))
    print("Test  R2 :", round(r2_score(y_test, pred_test2), 3))

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.3, random_state=42, stratify=y_cls)
    log = LogisticRegression(max_iter=1000)
    log.fit(X_train_c, y_train_c)
    pred_c = log.predict(X_test_c)
    print("\n(iii) Logistic Regression (Churn ~ Usage Frequency + Payment Delay)")
    print("Accuracy:", round(accuracy_score(y_test_c, pred_c), 3))

    print("\nOverfitting / Underfitting (short):")
    print("- Overfitting: good train score, poor test score.")
    print("- Underfitting: poor train and test scores.")


if __name__ == "__main__":
    main()

