import pandas as pd

from task._common import DEFAULT_CSV, load_df, numeric_summary


def main():
    df = load_df(DEFAULT_CSV)

    col = "Total Spend" if "Total Spend" in df.columns else df.select_dtypes(include="number").columns[0]
    stats = numeric_summary(df[col])

    print("\nColumn:", col)
    print("Mean:", round(stats["mean"], 3))
    print("Median:", round(stats["median"], 3))
    print("Std Dev:", round(stats["std"], 3))
    print("Skewness:", round(stats["skew"], 3))
    print("Kurtosis:", round(stats["kurt"], 3))

    print("\nInterpretation:")
    if abs(stats["skew"]) < 0.5:
        print("- Distribution looks approximately normal (low skew).")
    elif stats["skew"] > 0:
        print("- Distribution is right-skewed (positive skew).")
    else:
        print("- Distribution is left-skewed (negative skew).")


if __name__ == "__main__":
    main()

