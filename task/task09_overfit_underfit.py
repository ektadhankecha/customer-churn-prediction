import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from task._common import DEFAULT_CSV, load_df


def main():
    df = load_df(DEFAULT_CSV)
    X = df[["Usage Frequency"]].astype(float)
    y = df["Total Spend"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    for degree in [1, 2, 4]:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xtr = poly.fit_transform(X_train)
        Xte = poly.transform(X_test)

        model = LinearRegression()
        model.fit(Xtr, y_train)

        tr_pred = model.predict(Xtr)
        te_pred = model.predict(Xte)

        tr_mse = float(mean_squared_error(y_train, tr_pred))
        te_mse = float(mean_squared_error(y_test, te_pred))

        print(f"\nDegree {degree} polynomial regression")
        print("Train MSE:", round(tr_mse, 3))
        print("Test  MSE:", round(te_mse, 3))

    print("\nNotes:")
    print("- Higher degree can reduce train error but may increase test error (overfitting).")


if __name__ == "__main__":
    main()

