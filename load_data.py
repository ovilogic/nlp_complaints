import pandas as pd
from pathlib import Path

# path = Path(__file__).parent / "archive" / "complaints.csv"
# for i, chunk in enumerate(pd.read_csv(path, chunksize=100_000)):
#     chunk.to_parquet(f"complaints_{i}.parquet")
parent = Path(__file__).parent        
parquet_files = list((parent / "data" / "parquets").glob("complaints_*.parquet"))
parquet_merged = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
# print(type(parquet_merged))
load_csv = parquet_merged
print(load_csv.shape)
print(load_csv.columns)
# narrative = load_csv["Consumer complaint narrative"]
# print(narrative.head(100))

row = load_csv.loc[12, "narrative"]
print(row)
# print(load_csv.index[0])
# print(load_csv.loc[0, "text"])

# print(load_csv.iloc[1].astype(str, errors='ignore'))


