import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

c = sia.polarity_scores("I love this product!")
print(c)
print(type(c))
