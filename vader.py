import json

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from load_data import load_data, file_name, np

df = load_data(file_name)
sia = SentimentIntensityAnalyzer()

df["compound"] = df["narrative"].apply(
    lambda x: sia.polarity_scores(x)["compound"]
)
vader = {}
for product in df["product"].unique():
    product_df = df[df["product"] == product]
    vader[product] = float(np.round(product_df["compound"].mean(), 3))

with open("sentiment.json", "w") as file:
    json.dump(vader, file, indent=2)

print("VADER Sentiment Analysis Results:", vader)