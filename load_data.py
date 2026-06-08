import pandas as pd
from pathlib import Path

path = Path(__file__).parent / "data" / "complaints_processed.csv"
df = pd.read_csv(path, index_col=0)

parent = Path(__file__).parent        

# print(df.shape)
# print(df.columns)
# print(df.head())
# print(df.memory_usage(index=True, deep=True).sum() )
# row = df.loc[0, "narrative"]
# print(row)
# categories = df["product"].value_counts()
# print(categories)
# print(df.info())
# if df.shape[0] == len(df):
#     print(df.shape[0], df.shape[1], len(df))
# print(df.isnull().sum())
# print(df.isna().sum())
# print(df["product"].value_counts(normalize=True) * 100)

# print(df.duplicated())

df_clean = df.drop_duplicates(subset="narrative")
# print(df_clean.shape, df_clean.info(), sep="\n")

df = df_clean
df_non_null = df.dropna(subset=["narrative"])
# print(df_non_null.shape, df_non_null.info(), sep="\n")

df.dropna(inplace=True)
# print(df.info(), sep="\n")
# print(df.head())
print(df.columns)
print(df["product"].value_counts())
print(df["narrative"].str.len().describe())