import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SAMPLE_DATA_PATH = "data/sample/customer_churn_sample.csv"
OUTPUT_PATH = "data/graphs/correlation_outlier.png"


def main():
    os.makedirs("data/graphs", exist_ok=True)
    df = pd.read_csv(SAMPLE_DATA_PATH)

    # Encode categorical columns for correlation heatmap
    df_corr = df.copy()
    for col in df_corr.select_dtypes(include="object").columns:
        df_corr[col] = df_corr[col].astype("category").cat.codes

    plt.figure(figsize=(11, 4))

    plt.subplot(1, 2, 1)
    sns.heatmap(df_corr.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df["Total Spend"])
    plt.title("Outlier Check: Total Spend")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

