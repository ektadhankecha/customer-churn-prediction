import numpy as np
import pandas as pd

from task._common import DEFAULT_CSV, load_df


def main():
    df = load_df(DEFAULT_CSV)

    y = df["Total Spend"].astype(float)
    x1 = df["Usage Frequency"].astype(float)
    x2 = df["Payment Delay"].astype(float)

    print("\nDependent variable (Sales): Total Spend")
    print("Independent variables: Usage Frequency, Payment Delay")

    cov_x1 = float(np.cov(x1, y, ddof=1)[0, 1])
    cov_x2 = float(np.cov(x2, y, ddof=1)[0, 1])
    corr_x1 = float(np.corrcoef(x1, y)[0, 1])
    corr_x2 = float(np.corrcoef(x2, y)[0, 1])

    print("\nCovariance:")
    print("Usage Frequency vs Sales:", round(cov_x1, 3))
    print("Payment Delay vs Sales:", round(cov_x2, 3))

    print("\nCorrelation:")
    print("Usage Frequency vs Sales:", round(corr_x1, 3))
    print("Payment Delay vs Sales:", round(corr_x2, 3))


if __name__ == "__main__":
    main()

