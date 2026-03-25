import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SAMPLE_DATA_PATH = "data/sample/customer_churn_sample.csv"
OUTPUT_PATH = "data/graphs/categorical.png"


def main():
    os.makedirs("data/graphs", exist_ok=True)
    df = pd.read_csv(SAMPLE_DATA_PATH)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.countplot(x="Gender", data=df)
    plt.title("Gender Count")

    plt.subplot(1, 2, 2)
    sns.countplot(x="Contract Length", data=df)
    plt.title("Contract Length Count")
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

