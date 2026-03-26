import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from task._common import DEFAULT_CSV, OUT_DIR, ensure_out_dir, load_df


def main():
    ensure_out_dir()
    df = load_df(DEFAULT_CSV)

    print("\nMissing values per column:\n", df.isnull().sum())

    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].isnull().any():
            if df_filled[col].dtype == "object":
                df_filled[col] = df_filled[col].fillna(df_filled[col].mode(dropna=True)[0])
            else:
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    print("\nAfter handling missing values:\n", df_filled.isnull().sum())

    num_cols = df_filled.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        print("\nNo numeric columns for outlier detection.")
        return

    target_col = "Total Spend" if "Total Spend" in num_cols else num_cols[0]
    x = df_filled[target_col].astype(float)

    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = int(((x < low) | (x > high)).sum())

    print(f"\nOutlier check using IQR on '{target_col}':")
    print("Q1:", q1, "Q3:", q3, "IQR:", iqr)
    print("Lower bound:", low, "Upper bound:", high)
    print("Outliers count:", outliers)

    plt.figure(figsize=(6, 4))
    sns.boxplot(y=x)
    plt.title(f"Boxplot: {target_col} (Outliers)")
    plt.tight_layout()
    out_path = f"{OUT_DIR}/task03_outliers_boxplot.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved:", out_path)

    print("\nImpact of outliers:")
    print("- Outliers can increase mean and standard deviation.")
    print("- They can distort regression line and reduce model accuracy.")


if __name__ == "__main__":
    main()

