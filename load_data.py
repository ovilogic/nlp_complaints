import pandas as pd
import logging
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from pathlib import Path

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s - %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# path = Path(__file__).parent / "data" / "complaints_processed.csv"
# df = pd.read_csv(path, index_col=0)

# parent = Path(__file__).parent        

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

# df_clean = df.drop_duplicates(subset="narrative")
# # print(df_clean.shape, df_clean.info(), sep="\n")

# df = df_clean
# df_non_null = df.dropna(subset=["narrative"])
# print(df_non_null.shape, df_non_null.info(), sep="\n")

# df.dropna(inplace=True)
# # print(df.info(), sep="\n")
# # print(df.head())
# print(df.columns)
# print(df["product"].value_counts())
# print(df["narrative"].str.len().describe())

# Refactoring the code into functions
file_name = "complaints_processed.csv"
def load_data(file_name):
    logger.info(f"Loading data from {file_name}.")
    path = Path(__file__).parent / "data" / file_name
    df = pd.read_csv(path, index_col=0)
    # Initial inspection
    print("-" * 40, "Initial inspection", "-" * 40)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("Data types:")
    print(df.dtypes)
    print("First rows:")
    print(df.head(10))
    print("Last rows:")
    print(df.tail(10))
    print("Random rows:")
    print(df.sample(10))
    print("Info:")
    print(df.info())
    print("Summary stats for numeric columns:")
    print(df.describe())
    print("Summary stats for string columns:")
    print(df.describe(include="object"))
    print("Column names:")
    print(df.columns.tolist())
    df_clean = df.drop_duplicates(subset="narrative")
    df_clean.dropna(inplace=True)
    logger.info("Data cleaning completed.")
    print("\n" + "Returned cleaned DataFrame: ", df_clean.head())
    return df_clean

def text_preprocessing(series):
    logger.info("Starting text preprocessing.")
    series = series.str.strip()

    total_lower = (series == series.str.lower()).sum()
    logger.info(f"Total entries that are already lowercase: {total_lower} out of {len(series)}")

    punctuation = (series == series.str.contains(r"[^\w\s]", regex=True)).sum()
    logger.info(f"Total entries with punctuation: {punctuation} out of {len(series)}")
    
    # disabled_pipes = ["parser", "ner"]
    # nlp = spacy.load("en_core_web_sm", disable=disabled_pipes)
    # docs = nlp.pipe(series, batch_size=10)

    wordcount = series.str.split().str.len().sum()    
    total_stops = 0
    # stops = list(STOP_WORDS)
    stops_found = []  # To keep track of which stop words are found in the first few complaints
    for complaint in series:
        for word in complaint.split():
            if word in STOP_WORDS:
                total_stops += 1
                stops_found.append(word)
                # logger.debug(f"Found stop word: '{word}'")
    # logger.info(f"{total_stops} stop words found in all complaints: {stops_found}")
    logger.info(f"Total stop words found in all the complaints: {total_stops}.")
    logger.info(f"Total stop words found: {total_stops} out of {wordcount} total words.\
                This amounts to {(total_stops / wordcount) * 100:.2f}% of all words being stop words.")
    logger.info("Spacy tokenization completed.")


    return series

if __name__ == "__main__":
    df = load_data(file_name)
    print(text_preprocessing(df["narrative"]))

