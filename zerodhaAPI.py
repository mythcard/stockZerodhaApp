#!python
import argparse
from datetime import datetime, timedelta

from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split


# Function to fetch data
def fetch_data(kite, start_date, end_date, instrument_token, interval):
    return kite.historical_data(instrument_token, start_date, end_date, interval, False, True)


# Function to create sequences
def create_dataset(data, time_step=1, future_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - future_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step + future_step - 1, 0])
    return np.array(X), np.array(y)


# Function to build and compile LSTM model
def build_model(input_shape, features):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def main(api_key, access_token, api_secret, start_date, end_date, instrument_token, future_step, interval, time_step):
    kite = KiteConnect(api_key=api_key)
    previousAccessToken = access_token

    data = kite.generate_session(previousAccessToken, api_secret)
    kite.set_access_token(data["access_token"])

    features = ['open', 'high', 'low', 'volume']

    # Fetch data
    histData = fetch_data(kite, start_date, end_date, instrument_token, interval)

    # Convert to DataFrame and scale data
    df = pd.DataFrame(histData)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])

    # Prepare dataset
    X, y = create_dataset(scaled_data, time_step, future_step)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(features))
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], len(features))

    # Build and train the model
    model = build_model((time_step, len(features)), features)
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Predict
    last_time_step_data = scaled_data[-(time_step + future_step):]
    prediction_input = last_time_step_data[:time_step]
    prediction_input = prediction_input.reshape(1, time_step, len(features))
    predicted_price = model.predict(prediction_input)

    # Inverse transform and print
    dummy_array = np.zeros((predicted_price.shape[0], scaled_data.shape[1]))
    dummy_array[:, 0] = predicted_price.ravel()
    inverse_transformed = scaler.inverse_transform(dummy_array)
    final_predicted_price = inverse_transformed[:, 0]
    print(f"Predicted Close Price for {future_step}th minute:", final_predicted_price[0])

    # Get the current time
    current_time = datetime.now()

    # Calculate the time for the 10th minute from now
    time_10th_minute_from_now = current_time + timedelta(minutes=10)

    # Print the calculated time
    print("Time for the 10th minute from now:", time_10th_minute_from_now.strftime('%Y-%m-%d %H:%M:%S'))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True, help="Kite Connect API key")
    parser.add_argument("--access_token", type=str, required=True, help="Kite Connect Access Token")
    parser.add_argument("--api_secret", type=str, required=True, help="Kite API secret")
    parser.add_argument("--start_date", type=str, required=True,
                        help="Start date for historical data in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, required=True, help="End date for historical data in YYYY-MM-DD format")
    parser.add_argument("--instrument_token", type=str, required=True, help="Instrument token for the stock")
    parser.add_argument("--future_step", type=int, choices=[5, 10, 30, 60], required=True,
                        help="Prediction minute (5, 10, 30, 60)")
    parser.add_argument("--interval", type=str, default="minute", help="Interval for historical data")
    parser.add_argument("--time_step", type=int, default=100, help="Number of time steps for the LSTM model")

    args = parser.parse_args()

    main(args.api_key, args.access_token, args.api_secret, args.start_date, args.end_date, args.instrument_token,
         args.future_step,
         args.interval, args.time_step)



# print("Predicted Close Price for 10th minute:", final_predicted_price[0])


