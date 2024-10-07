import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import os

def fetch_bitcoin_data(start_date, end_date):
    """
    Fetch Bitcoin price data from Yahoo Finance
    """
    btc_data = yf.download("BTC-USD", start=start_date, end=end_date)
    btc_data = btc_data.reset_index()
    btc_data = btc_data[["Date", "Close"]]
    btc_data.columns = ["Date", "Price"]
    btc_data["Date"] = pd.to_datetime(btc_data["Date"]).dt.date  # Convert to date
    return btc_data

def fetch_news_data(start_date, end_date):
    """
    Fetch news data related to Bitcoin from NewsAPI
    """
    newsapi = NewsApiClient(api_key=os.environ['NEWSAPI_KEY'])
    
    # Set the earliest allowed date to 30 days ago
    earliest_allowed_date = datetime.now().date() - timedelta(days=30)
    adjusted_start_date = max(start_date, earliest_allowed_date)
    
    all_articles = []
    current_date = adjusted_start_date
    
    while current_date <= end_date:
        try:
            articles = newsapi.get_everything(q="bitcoin",
                                              from_param=current_date.strftime("%Y-%m-%d"),
                                              to=min(current_date + timedelta(days=7), end_date).strftime("%Y-%m-%d"),
                                              language="en",
                                              sort_by="publishedAt")
            
            all_articles.extend(articles["articles"])
        except Exception as e:
            print(f"Error fetching news data: {str(e)}")
            break
        
        current_date += timedelta(days=7)
    
    if not all_articles:
        return pd.DataFrame(columns=["Date", "title", "description"])
    
    news_data = pd.DataFrame(all_articles)
    news_data["Date"] = pd.to_datetime(news_data["publishedAt"]).dt.date
    news_data = news_data[["Date", "title", "description"]]
    return news_data
