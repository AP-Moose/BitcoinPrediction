from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def prepare_data(data, target_col, features, lag=1):
    """
    Prepare data for machine learning model
    """
    for feature in features:
        data[f"{feature}_lag_{lag}"] = data[feature].shift(lag)
    
    data = data.dropna()
    
    X = data[[f"{feature}_lag_{lag}" for feature in features]]
    y = data[target_col]
    
    return X, y

def train_model(data):
    """
    Train a Random Forest Regressor model
    """
    features = ["Price", "sentiment_score"]
    X, y = prepare_data(data, "Price", features)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def make_prediction(model, scaler, data, prediction_period):
    """
    Make predictions using the trained model
    """
    features = ["Price", "sentiment_score"]
    X, _ = prepare_data(data, "Price", features)
    
    X_scaled = scaler.transform(X)
    
    last_date = data["Date"].iloc[-1]
    
    if prediction_period == "Daily":
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    elif prediction_period == "Weekly":
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=12, freq="W")
    else:  # Monthly
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=6, freq="M")
    
    predictions = []
    
    for _ in future_dates:
        prediction = model.predict(X_scaled[-1].reshape(1, -1))[0]
        predictions.append(prediction)
        
        new_row = np.array([prediction, X_scaled[-1, 1]])  # Assume sentiment remains the same
        X_scaled = np.vstack([X_scaled, new_row])
    
    return predictions
