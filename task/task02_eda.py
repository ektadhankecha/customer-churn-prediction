import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from task._common import DEFAULT_CSV, OUT_DIR, detect_cols, ensure_out_dir, load_df, require


def main():
    ensure_out_dir()
    df = load_df(DEFAULT_CSV)
    cols = detect_cols(df)

    sales = cols.sales
    quantity = "Usage Frequency" if "Usage Frequency" in df.columns else cols.quantity
    discount = "Payment Delay" if "Payment Delay" in df.columns else cols.discount

    require(df, {"sales": sales, "quantity": quantity, "discount": discount})

    profit = (df[sales] * 0.25) - (df[discount] * 2.0)

    print("\nUnivariate Analysis (using proxies)")
    print("Sales ->", sales)
    print("Profit -> derived: (Sales*0.25) - (PaymentDelay*2)")
    print("Quantity ->", quantity)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    sns.histplot(df[sales], kde=True)
    plt.title("Sales (Total Spend) - Histogram")

    plt.subplot(2, 2, 2)
    sns.boxplot(y=df[sales])
    plt.title("Sales (Total Spend) - Boxplot")

    plt.subplot(2, 2, 3)
    sns.histplot(profit, kde=True, color="orange")
    plt.title("Profit (Derived) - Histogram")

    plt.subplot(2, 2, 4)
    sns.boxplot(y=profit, color="orange")
    plt.title("Profit (Derived) - Boxplot")
    plt.tight_layout()
    out1 = f"{OUT_DIR}/task02_univariate.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print("\nSaved:", out1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=df[sales], y=profit)
    plt.title("Sales vs Profit")

    plt.subplot(1, 3, 2)
    sns.scatterplot(x=df[discount], y=profit)
    plt.title("Payment Delay vs Profit")

    plt.subplot(1, 3, 3)
    sns.scatterplot(x=df[quantity], y=df[sales])
    plt.title("Usage Frequency vs Sales")
    plt.tight_layout()
    out2 = f"{OUT_DIR}/task02_bivariate.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print("Saved:", out2)

    corr_df = pd.DataFrame(
        {
            "Sales": pd.to_numeric(df[sales], errors="coerce"),
            "Profit": pd.to_numeric(profit, errors="coerce"),
            "Quantity": pd.to_numeric(df[quantity], errors="coerce"),
            "Discount": pd.to_numeric(df[discount], errors="coerce"),
        }
    ).dropna()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    out3 = f"{OUT_DIR}/task02_correlation.png"
    plt.savefig(out3, dpi=150)
    plt.close()
    print("Saved:", out3)


if __name__ == "__main__":
    main()

