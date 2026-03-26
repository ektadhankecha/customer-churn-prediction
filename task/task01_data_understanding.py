import pandas as pd

from task._common import DEFAULT_CSV, load_df


def main():
    df = load_df(DEFAULT_CSV)

    print("\nFirst 5 rows:\n", df.head())
    print("\nLast 5 rows:\n", df.tail())
    print("\nShape:", df.shape)
    print("\nColumns:\n", list(df.columns))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    discrete = [c for c in numeric_cols if pd.api.types.is_integer_dtype(df[c])]
    continuous = [c for c in numeric_cols if c not in discrete]

    nominal = cat_cols
    ordinal = []

    if "Contract Length" in nominal:
        nominal.remove("Contract Length")
        ordinal.append("Contract Length")

    print("\nQuantitative (Discrete):", discrete)
    print("Quantitative (Continuous):", continuous)
    print("Qualitative (Nominal):", nominal)
    print("Qualitative (Ordinal):", ordinal)


if __name__ == "__main__":
    main()

