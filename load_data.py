import pandas as pd
from pathlib import Path

path = Path(__file__).parent / "data" / "complaints_processed.csv"
df = pd.read_csv(path)
# for i, chunk in enumerate(pd.read_csv(path, chunksize=100_000)):
#     chunk.to_parquet(f"complaints_{i}.parquet")

parent = Path(__file__).parent        
# parquet_files = list((parent / "data" / "parquets").glob("complaints_*.parquet"))
# load_parquets= pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
# print(load_parquets.shape)
# print(load_parquets.columns)
# # print(load_parquets.head())
# row = load_parquets.iloc[0]
# print(row)

# print(load_parquets.memory_usage(index=True, deep=True))

# sample_df = load_parquets.sample(frac=0.1, random_state=42)

# sample_df.to_parquet(parent / "data" / "sample_complaints.parquet", index=True)
# index=True will include the index in the parquet file, which can be useful for preserving the original row numbers. If you don't need the index, you can set it to False. 
# load_sample = pd.read_parquet(parent / "data" / "sample_complaints.parquet")
# print(load_sample.shape)
# print(load_sample.columns)
# print(load_sample.head())

print(df.shape)
print(df.columns)
print(df.head())
print(df.memory_usage(index=True, deep=True).sum() / (1024 * 1024), "MB")