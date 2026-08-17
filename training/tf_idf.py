from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love Python",
    "Python is great",
    "I love programming"
]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
print(tfidf_matrix.toarray())
print(vectorizer.get_feature_names_out())