import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from task._common import DEFAULT_CSV, OUT_DIR, ensure_out_dir, load_df


def main():
    ensure_out_dir()
    df = load_df(DEFAULT_CSV)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df["Total Spend"], kde=True)
    plt.title("Univariate: Total Spend")

    plt.subplot(1, 2, 2)
    sns.countplot(x="Contract Length", hue="Churn", data=df)
    plt.title("Bi-variate: Contract Length vs Churn")
    plt.xticks(rotation=20)
    plt.tight_layout()
    out1 = f"{OUT_DIR}/task13_uni_bi.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print("\nSaved:", out1)

    df_corr = df.copy()
    for col in df_corr.select_dtypes(include="object").columns:
        df_corr[col] = df_corr[col].astype("category").cat.codes

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_corr.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title("Multi-variate: Heatmap (encoded)")
    plt.tight_layout()
    out2 = f"{OUT_DIR}/task13_heatmap.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print("Saved:", out2)

    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df["Total Spend"])
    plt.title("Normalization/Skew check: Total Spend")
    plt.tight_layout()
    out3 = f"{OUT_DIR}/task13_outliers.png"
    plt.savefig(out3, dpi=150)
    plt.close()
    print("Saved:", out3)


if __name__ == "__main__":
    main()

