import numpy as np
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
# print(tfidf_matrix.shape)  # This will print the shape sparse matrix representation of the TF-IDF values
# print(tfidf_matrix.toarray())  # This will print the dense representation of the TF-IDF values

# print(tfidf_matrix.indptr)
# print(tfidf_matrix.data)
# print(vectorizer.get_feature_names_out())  # This will print the feature names (terms)\ 
# corresponding to the columns of the TF-IDF matrix
# print(tfidf_matrix.indices[1])
mask1 = (tfidf_matrix.indices ==  1)
data1 = tfidf_matrix.data[mask1]
print(data1)
mask1_where = np.where(mask1)
data1_where = tfidf_matrix.data[mask1_where]
print(data1 == data1_where)

# Does where work without a condition? Yes, it returns all indices of the array.
# print(np.where(np.array([True, False, True]))) # But if you pass a Boolean array, it returns the indices of the True values.