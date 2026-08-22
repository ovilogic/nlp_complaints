from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love Python",
    "Python is great",
    "I love programming"
]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
# print(tfidf_matrix.toarray())
# print(vectorizer.get_feature_names_out())

# Sparse matrix returned
print(tfidf_matrix.shape)  # This will print the shape sparse matrix representation of the TF-IDF values
print(tfidf_matrix.toarray())  # This will print the dense representation of the TF-IDF values
print(tfidf_matrix.indices)
print(tfidf_matrix.indptr)
print(tfidf_matrix.data)