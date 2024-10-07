import pandas as pd
from textblob import TextBlob

def analyze_sentiment(news_data):
    """
    Analyze sentiment of news data using TextBlob
    """
    def get_sentiment_score(text):
        if pd.isna(text):
            return 0
        return TextBlob(str(text)).sentiment.polarity
    
    news_data["title_sentiment"] = news_data["title"].apply(get_sentiment_score)
    news_data["description_sentiment"] = news_data["description"].apply(get_sentiment_score)
    
    news_data["sentiment_score"] = (news_data["title_sentiment"] + news_data["description_sentiment"]) / 2
    
    sentiment_scores = news_data.groupby("Date")["sentiment_score"].mean().reset_index()
    return sentiment_scores
