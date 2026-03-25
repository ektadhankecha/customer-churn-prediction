import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SAMPLE_DATA_PATH = "data/sample/customer_churn_sample.csv"
OUTPUT_PATH = "data/graphs/univariate.png"


def main():
    os.makedirs("data/graphs", exist_ok=True)
    df = pd.read_csv(SAMPLE_DATA_PATH)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.histplot(df["Total Spend"], kde=True, bins=10, color="#3498db")
    plt.title("Distribution of Total Spend")

    plt.subplot(1, 2, 2)
    sns.histplot(df["Tenure"], kde=True, bins=10, color="#2ecc71")
    plt.title("Distribution of Tenure")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

