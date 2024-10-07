import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import traceback

from data_fetcher import fetch_crypto_data, fetch_news_data
from sentiment_analyzer import analyze_sentiment
from ml_model import train_model, make_prediction
from visualizer import plot_crypto_price, plot_sentiment_trend, plot_prediction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPPORTED_CRYPTOCURRENCIES = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Litecoin": "LTC",
    "Ripple": "XRP",
    "Dogecoin": "DOGE"
}

def main():
    try:
        st.set_page_config(page_title="Cryptocurrency Price Predictor", layout="wide")

        st.title("Cryptocurrency Price Predictor")

        # Sidebar for user input
        st.sidebar.header("User Input")
        selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(SUPPORTED_CRYPTOCURRENCIES.keys()))
        prediction_period = st.sidebar.selectbox("Select Prediction Period", ["Daily", "Weekly", "Monthly"])
        model_type = st.sidebar.selectbox("Select Model Type", ["Random Forest", "LSTM"])

        # Set the default start date to 30 days ago
        default_start_date = datetime.now() - timedelta(days=30)
        custom_start_date = st.sidebar.date_input("Select Start Date", default_start_date)
        custom_end_date = st.sidebar.date_input("Select End Date", datetime.now())

        st.sidebar.warning("Note: Due to API limitations, news data can only be fetched for the last 30 days.")

        # Fetch and process data
        crypto_symbol = SUPPORTED_CRYPTOCURRENCIES[selected_crypto]
        crypto_data = fetch_crypto_data(crypto_symbol, custom_start_date, custom_end_date)
        news_data = fetch_news_data(selected_crypto, custom_start_date, custom_end_date)

        if news_data.empty:
            st.error("Unable to fetch news data. Please try again later or adjust the date range.")
        else:
            # Analyze sentiment
            sentiment_scores = analyze_sentiment(news_data)

            # Combine price and sentiment data
            combined_data = pd.merge(crypto_data, sentiment_scores, on="Date", how="inner")

            # Train model and make predictions
            model_type_param = "rf" if model_type == "Random Forest" else "lstm"
            
            model, scaler = train_model(combined_data, model_type=model_type_param)
            predictions = make_prediction(model, scaler, combined_data, prediction_period, model_type=model_type_param)

            # Display results
            st.header(f"{selected_crypto} Price Analysis and Prediction")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"Historical {selected_crypto} Price")
                fig_price = plot_crypto_price(crypto_data, selected_crypto)
                st.plotly_chart(fig_price, use_container_width=True)

            with col2:
                st.subheader("Sentiment Trend")
                fig_sentiment = plot_sentiment_trend(sentiment_scores)
                st.plotly_chart(fig_sentiment, use_container_width=True)

            st.subheader(f"{selected_crypto} Price Prediction ({prediction_period}) - {model_type} Model")
            fig_prediction = plot_prediction(combined_data, predictions, prediction_period, selected_crypto)
            st.plotly_chart(fig_prediction, use_container_width=True)

            st.write(f"Predicted {selected_crypto} price for next {prediction_period.lower()}: ${predictions[-1]:.2f}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please try again or contact support if the issue persists.")

if __name__ == '__main__':
    main()
