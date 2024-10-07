import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data_fetcher import fetch_bitcoin_data, fetch_news_data
from sentiment_analyzer import analyze_sentiment
from ml_model import train_model, make_prediction
from visualizer import plot_bitcoin_price, plot_sentiment_trend, plot_prediction

st.set_page_config(page_title="Bitcoin Price Predictor", layout="wide")

st.title("Bitcoin Price Predictor")

# Sidebar for user input
st.sidebar.header("User Input")
prediction_period = st.sidebar.selectbox("Select Prediction Period", ["Daily", "Weekly", "Monthly"])
model_type = st.sidebar.selectbox("Select Model Type", ["Random Forest", "LSTM"])

# Set the default start date to 30 days ago
default_start_date = datetime.now() - timedelta(days=30)
custom_start_date = st.sidebar.date_input("Select Start Date", default_start_date)
custom_end_date = st.sidebar.date_input("Select End Date", datetime.now())

st.sidebar.warning("Note: Due to API limitations, news data can only be fetched for the last 30 days.")

# Fetch and process data
btc_data = fetch_bitcoin_data(custom_start_date, custom_end_date)
news_data = fetch_news_data(custom_start_date, custom_end_date)

if news_data.empty:
    st.error("Unable to fetch news data. Please try again later or adjust the date range.")
else:
    # Analyze sentiment
    sentiment_scores = analyze_sentiment(news_data)

    # Combine price and sentiment data
    combined_data = pd.merge(btc_data, sentiment_scores, on="Date", how="inner")

    # Train model and make predictions
    model_type_param = "rf" if model_type == "Random Forest" else "lstm"
    model, scaler = train_model(combined_data, model_type=model_type_param)
    predictions = make_prediction(model, scaler, combined_data, prediction_period, model_type=model_type_param)

    # Display results
    st.header("Bitcoin Price Analysis and Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Historical Bitcoin Price")
        fig_price = plot_bitcoin_price(btc_data)
        st.plotly_chart(fig_price, use_container_width=True)

    with col2:
        st.subheader("Sentiment Trend")
        fig_sentiment = plot_sentiment_trend(sentiment_scores)
        st.plotly_chart(fig_sentiment, use_container_width=True)

    st.subheader(f"Bitcoin Price Prediction ({prediction_period}) - {model_type} Model")
    fig_prediction = plot_prediction(combined_data, predictions, prediction_period)
    st.plotly_chart(fig_prediction, use_container_width=True)

    st.write(f"Predicted Bitcoin price for next {prediction_period.lower()}: ${predictions[-1]:.2f}")
