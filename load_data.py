import pandas as pd
from pathlib import Path

# path = Path(__file__).parent / "archive" / "complaints.csv"
# for i, chunk in enumerate(pd.read_csv(path, chunksize=100_000)):
#     chunk.to_parquet(f"complaints_{i}.parquet")
parent = Path(__file__).parent        
parquet_files = list((parent / "data" / "parquets").glob("complaints_*.parquet"))
load_parquets= pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
print(load_parquets.shape)
print(load_parquets.columns)
print(load_parquets.head())



