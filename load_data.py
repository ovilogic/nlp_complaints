import pandas as pd
import logging
import spacy
import numpy as np
import logging
from spacy.lang.en.stop_words import STOP_WORDS
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

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
    print(df.describe(include="str"))
    print("Column names:")
    print(df.columns.tolist())
    df_clean = df.drop_duplicates(subset="narrative")
    df_clean.dropna(inplace=True)
    logger.info("Data cleaning completed.\n" + "-" * 80)
    return df_clean

def data_auditing(series):
    logger.info("Starting data auditing.")
    series = series.str.strip()
   
    total_lower = (series == series.str.lower()).sum()
    logger.info(f"Total entries that are already lowercase: {total_lower} out of {len(series)} = {(total_lower / len(series)) * 100:.2f}%")
    punctuation = series.str.contains(r"[^\w\s]", regex=True).sum()
    logger.info(f"Total entries with punctuation: {punctuation} out of {len(series)} = {(punctuation / len(series)) * 100:.2f}%")
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
    # logger.info("*" * 80 + f"testing exploded: {exploded.tail(30)}")
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
    logger.info(f"Total stop words found: {total_stops} out of {word_count.sum()} total words.\
                This amounts to {(total_stops / word_count.sum()) * 100:.2f}% of all words being stop words.")
    
    return series
'''
Better TF-IDF weights. Your core deliverable is per-category TF-IDF. A term like "charge" is genuinely important in credit_card complaints — but "charged", "charges", "charging" spread its frequency across variants. Lemmatized, that frequency consolidates onto one token, giving it a stronger and more accurate IDF weight.
Cleaner VADER input. VADER works on raw text, not lemmatized text — so lemmatization won't affect your sentiment sparse_matrix directly. But when you're interpreting which words drive frustration intensity, having normalized tokens makes those patterns more legible.
'''
def lemmatization(df, column_name):
    df = df.copy()
    nlp = spacy.load('en_core_web_sm')
    logger.info("Model loaded.")
    series = df[column_name]
    df["Lemmatized"] = ""  # Initialize the new column with empty strings
    # lemmatized_texts = []
    # for doc in nlp.pipe(series, batch_size=1000):
    #     lemmatized_doc = " ".join(token.lemma_ for token in doc)
    #     # lemmatized_texts.append(lemmatized_doc)
    #     df["Lemmatized"] = lemmatized_doc
    with nlp.select_pipes(disable=["parser", "ner"]):
        for idx, doc in enumerate(nlp.pipe(series, batch_size=1000)):
            df.iloc[idx, df.columns.get_loc("Lemmatized")] = " ".join(token.lemma_ for token in doc)
            if (idx + 1) % 10000 == 0:
                logger.info(f"Processed {idx + 1} documents")

    logger.info("Lemmatisation is now complete. Proceeding to saving as a parquet and sampling.")
    df.to_parquet("df_lemmatized.parquet")
    # Sample check the first few lemmatized entries
    # sample = df[["narrative", "Lemmatized"]].head(10)
    # for _, row in sample.iterrows():
    #     print("-" * 10)
    #     print(f"Original: {row['narrative'][:300]}...")  # Print the first 300 characters for brevity
    #     print(f"Lemmatized: {row['Lemmatized'][:300]}...")
    return df

def calculate_tfidf(text_series):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_series)
    return tfidf_matrix, vectorizer.get_feature_names_out()

if __name__ == "__main__":
    df = load_data(file_name)
    # data_auditing(df["narrative"])
    # # lemmatized_df = lemmatization(df, "narrative")
    load_lemmatized_df = pd.read_parquet("./lemmatized_already/df_lemmatized.parquet")

    tfidf = calculate_tfidf(load_lemmatized_df["Lemmatized"])
    terms = tfidf[1]  # Get the feature names (terms)
    sparse_matrix = tfidf[0] # Remmember, this is a sparse matrix.

    for product in df["product"].unique():
        # First, let's get a mask (single Series of boolean values). As the notes say,
        # this can be used directly as a mask for filtering.
        product_rows = df["product"].eq(product)
        product_row_indices = product_rows.to_numpy().nonzero()[0] # The whole point of going back to the original dense array \
        # to get the row numbers (indices). You can't really get indices from a sparse, non in the usual way.
        # print(product, product_row_indices, end="-" * 40 + "\n")
        product_tfidf = sparse_matrix[product_row_indices] # but you can pass a Boolean mask to a sparse and that's okay.
        means_of_all_terms = []
        # print("Max column index in product_tfidf:", product_tfidf.indices.max())
        # print("len(terms) - 1:", len(terms) - 1)
        for i in range(len(terms)):
            if i == 10:
                term_word = terms[i]
                print(term_word)
                mask_a = product_tfidf.indices == i
                values_a = product_tfidf.data[mask_a]
                print(values_a)
        #     # finds every position (across all rows of product_tfidf) \ 
        #     # where the stored value belongs to column i, i.e. to term i.
        #     term_indices = np.where(product_tfidf.indices == i) # this produces a mask
        #     print("Full matrix shape:", sparse_matrix.shape)
        #     print("Product matrix shape:", product_tfidf.shape)
        #     print("Same number of columns?", sparse_matrix.shape[1] == product_tfidf.shape[1])
        #     print("Number of terms:", len(terms))
        #     term_data = product_tfidf.data[term_indices] # Using that mask on product_tfidf.data pulls out every nonzero TF-IDF value for term i, 
            # across every document belonging to that product — 
            # which is exactly what you want if you're computing something like a per-term average for that product.
        #     term_mean_tfidf = term_data.sum() / len(product_rows[product_rows == True]) 
        #     means_of_all_terms.append(term_mean_tfidf)
        # top_10_terms = sorted(zip(terms, means_of_all_terms), key=lambda x: x[1], reverse=True)[:10]
        # print(top_10_terms, end="-" * 40 + "\n")



