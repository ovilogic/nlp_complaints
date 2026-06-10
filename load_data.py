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
    # print("\n" + "Returned cleaned DataFrame: ", df_clean.head())
    return df_clean

def data_auditing(series):
    logger.info("Starting data auditing.")
    series = series.str.strip()

    total_lower = (series == series.str.lower()).sum()
    logger.info(f"Total entries that are already lowercase: {total_lower} out of {len(series)}")
    punctuation = series.str.contains(r"[^\w\s]", regex=True).sum()
    logger.info(f"Total entries with punctuation: {punctuation} out of {len(series)}")
  
    word_lists = series.str.split()
    unique_words = []
    # Vocabulary size:
    for word_list in word_lists:
        unique_complaint = set(word_list)
        unique_words.extend(unique_complaint)
    unique_words_count = len(set(unique_words))
    logger.info(f"Vocabulary size (unique words): {unique_words_count}")
    # Most frequent words:
    exploded = word_lists.explode()
    logger.info(f"Total words (including duplicates): {len(exploded)}")
    word_freq = exploded.value_counts()
    most_frequent = word_freq.head(10)
    logger.info(f"Most frequent 10 words: \n{most_frequent}")
    # Rare words:
    rare_words = (word_freq == 1).sum()
    logger.info(f"Words appearing once: \n{rare_words}"
                f"\n Which amounts to {(rare_words / unique_words_count) * 100:.2f}% of the vocabulary.")

    word_count = word_lists.str.len() # Number of words in each complaint
    logger.info(f"Complaint length distribution (in words): \n {word_count.describe()}")
    logger.info(f"95th percentile complaint length: {word_count.quantile(0.95):.0f} words")

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
    logger.info(f"Total stop words found: {total_stops} out of {word_count.sum()} total words.\
                This amounts to {(total_stops / word_count.sum()) * 100:.2f}% of all words being stop words.")
    


    return series

if __name__ == "__main__":
    df = load_data(file_name)
    data_auditing(df["narrative"])

