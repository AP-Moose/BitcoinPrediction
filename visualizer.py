import plotly.graph_objects as go
import pandas as pd

def plot_crypto_price(crypto_data, crypto_name):
    """
    Plot historical cryptocurrency price
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=crypto_data["Date"], y=crypto_data["Price"], mode="lines", name=f"{crypto_name} Price"))
    fig.update_layout(title=f"Historical {crypto_name} Price", xaxis_title="Date", yaxis_title="Price (USD)")
    return fig

def plot_sentiment_trend(sentiment_scores):
    """
    Plot sentiment trend
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sentiment_scores["Date"], y=sentiment_scores["sentiment_score"], mode="lines", name="Sentiment Score"))
    fig.update_layout(title="Sentiment Trend", xaxis_title="Date", yaxis_title="Sentiment Score")
    return fig

def plot_prediction(data, predictions, prediction_period, crypto_name):
    """
    Plot historical data and predictions
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Price"], mode="lines", name="Historical Price"))
    
    # Predictions
    last_date = data["Date"].iloc[-1]
    
    if prediction_period == "Daily":
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    elif prediction_period == "Weekly":
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=12, freq="W")
    else:  # Monthly
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=6, freq="M")
    
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode="lines", name="Predicted Price", line=dict(dash="dash")))
    
    fig.update_layout(title=f"{crypto_name} Price Prediction ({prediction_period})", xaxis_title="Date", yaxis_title="Price (USD)")
    return fig
