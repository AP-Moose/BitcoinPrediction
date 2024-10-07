import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta

def fetch_bitcoin_data(start_date, end_date):
    """
    Fetch Bitcoin price data from Yahoo Finance
    """
    btc_data = yf.download("BTC-USD", start=start_date, end=end_date)
    btc_data = btc_data.reset_index()
    btc_data = btc_data[["Date", "Close"]]
    btc_data.columns = ["Date", "Price"]
    return btc_data

def fetch_news_data(start_date, end_date):
    """
    Fetch news data related to Bitcoin from NewsAPI
    """
    newsapi = NewsApiClient(api_key="YOUR_NEWSAPI_KEY")  # Replace with your actual NewsAPI key
    
    all_articles = []
    current_date = start_date
    
    while current_date <= end_date:
        articles = newsapi.get_everything(q="bitcoin",
                                          from_param=current_date.strftime("%Y-%m-%d"),
                                          to=min(current_date + timedelta(days=30), end_date).strftime("%Y-%m-%d"),
                                          language="en",
                                          sort_by="publishedAt")
        
        all_articles.extend(articles["articles"])
        current_date += timedelta(days=30)
    
    news_data = pd.DataFrame(all_articles)
    news_data["Date"] = pd.to_datetime(news_data["publishedAt"]).dt.date
    news_data = news_data[["Date", "title", "description"]]
    return news_data

