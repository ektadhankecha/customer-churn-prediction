import pandas as pd

from task._common import DEFAULT_CSV, load_df


def eda_report(df: pd.DataFrame):
    print("\nInfo:")
    df.info()

    print("\nDescribe (numeric):\n", df.describe())
    print("\nMissing values:\n", df.isnull().sum())

    num = df.select_dtypes(include="number")
    if not num.empty:
        print("\nCorrelation:\n", num.corr(numeric_only=True))


def main():
    df = load_df(DEFAULT_CSV)
    eda_report(df)


if __name__ == "__main__":
    main()

