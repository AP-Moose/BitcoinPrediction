from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

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

def create_sequences(X, y, time_steps=1):
    """
    Create sequences for LSTM model
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_model(data, model_type="rf"):
    """
    Train a Random Forest Regressor or LSTM model
    """
    features = ["Price", "sentiment_score"]
    X, y = prepare_data(data, "Price", features)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
    elif model_type == "lstm":
        X_train_seq, y_train_seq = create_sequences(X_train, y_train)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test)
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    else:
        raise ValueError("Invalid model type. Choose 'rf' or 'lstm'.")
    
    return model, scaler

def make_prediction(model, scaler, data, prediction_period, model_type="rf"):
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
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=6, freq="ME")
    
    predictions = []
    
    if model_type == "rf":
        for _ in future_dates:
            prediction = model.predict(X_scaled[-1].reshape(1, -1))[0]
            predictions.append(prediction)
            
            new_row = np.array([prediction, X_scaled[-1, 1]])  # Assume sentiment remains the same
            X_scaled = np.vstack([X_scaled, new_row])
    elif model_type == "lstm":
        X_seq, _ = create_sequences(pd.DataFrame(X_scaled), pd.Series(np.zeros(len(X_scaled))))
        for _ in future_dates:
            prediction = model.predict(X_seq[-1:])
            predictions.append(prediction[0][0])
            
            new_row = np.array([[prediction[0][0], X_scaled[-1, 1]]])  # Assume sentiment remains the same
            X_scaled = np.vstack([X_scaled, new_row])
            X_seq = np.vstack([X_seq[1:], new_row.reshape(1, 1, 2)])
    
    return predictions
