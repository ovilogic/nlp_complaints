import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# documents = [
#     "I love Python",
#     "Python is great",
#     "I love programming"
# ]

# vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = vectorizer.fit_transform(documents)
# print(tfidf_matrix.toarray())
# print(vectorizer.get_feature_names_out())

# Sparse matrix returned
# print(tfidf_matrix.shape)  # This will print the shape sparse matrix representation of the TF-IDF values
# print(tfidf_matrix.toarray())  # This will print the dense representation of the TF-IDF values

# print(tfidf_matrix.indptr)
# print(tfidf_matrix.data)
# print(vectorizer.get_feature_names_out())  # This will print the feature names (terms)\ 
# corresponding to the columns of the TF-IDF matrix
# print(tfidf_matrix.indices[1])
# mask1 = (tfidf_matrix.indices ==  1)
# data1 = tfidf_matrix.data[mask1]
# print(data1)
# mask1_where = np.where(mask1)
# data1_where = tfidf_matrix.data[mask1_where]
# print(data1 == data1_where)

# Does where work without a condition? Yes, it returns all indices of the array.
# print(np.where(np.array([True, False, True]))) # But if you pass a Boolean array, 
# it returns the indices of the True values.

# Exercise: testing results for global IDF vs local IDF. 
# The global IDF is computed across all documents, 
# while the local IDF is computed for a specific subset of documents.
'''
Let's build a tiny corpus — 3 categories, 
2 documents each, so 6 documents total. 
Focus word: "overdraft".
Corpus:

Retail banking:

R1: "overdraft fee charged twice"
R2: "overdraft overdraft again this month"

Credit card:

C1: "interest rate too high"
C2: "late fee and interest charge"

Mortgages:

M1: "escrow account confusing"
M2: "interest rate on my mortgage"
'''
dict_corpus = {
    "Retail banking": [
        "overdraft fee charged twice",
        "overdraft overdraft again this month"
    ],
    "Credit card": [
        "interest rate too high",
        "late fee and interest charge"
    ],
    "Mortgages": [
        "escrow account confusing",
        "interest rate on my mortgage"
    ]
}

df = pd.DataFrame(dict_corpus)
# stacked = df.stack()
# print(stacked.index)
df = df.melt()
df.columns = ["product", "corpus"]
print(df)

vectorizer = TfidfVectorizer(norm=None)
tfidf_matrix = vectorizer.fit_transform(df["corpus"])
dense_tfidf = tfidf_matrix.toarray()
print(dense_tfidf[:, 15:])
terms = vectorizer.get_feature_names_out()
# print(np.where(terms == "overdraft"))
# for i, j in enumerate(terms):
#     print(i, j)

# print(dense_tfidf[:, 15])
# print(tfidf_matrix.indices)
# pretty_dense = pd.DataFrame(dense_tfidf[:], columns=terms)
# print(pretty_dense.iloc[:, 15])

# Not going to write it, but a local IDF (for a specific subset of documents)
# for 'Retail banking', will be 0 for a term like 'overdraft', as it appears in all of (2) the 
# documents (0 and 1). Meaning the entire tfidf score for 'overdraft' will be 0.
