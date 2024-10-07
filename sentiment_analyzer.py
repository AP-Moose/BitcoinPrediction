import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

nltk.download("vader_lexicon")

def analyze_sentiment(news_data):
    """
    Analyze sentiment of news data using NLTK's VADER sentiment analyzer
    """
    sia = SentimentIntensityAnalyzer()
    
    def get_sentiment_score(text):
        return sia.polarity_scores(text)["compound"]
    
    news_data["title_sentiment"] = news_data["title"].apply(get_sentiment_score)
    news_data["description_sentiment"] = news_data["description"].apply(get_sentiment_score)
    
    news_data["sentiment_score"] = (news_data["title_sentiment"] + news_data["description_sentiment"]) / 2
    
    sentiment_scores = news_data.groupby("Date")["sentiment_score"].mean().reset_index()
    return sentiment_scores

