#!python
import logging
from kiteconnect import KiteConnect

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)

kite = KiteConnect()

# Redirect the user to the login url obtained
# from kite.login_url(), and receive the request_token
# from the registered redirect url after the login flow.
# Once you have the request_token, obtain the access_token
# as follows.

previousAccessToken = ""

data = kite.generate_session(previousAccessToken, api_secret="")
kite.set_access_token(data["access_token"])

# Get instruments
data = kite.instruments()

# print(type(data))
# print(" Instrument data: ")
# print(data)

histData = kite.historical_data('5195009', '2023-12-11', "2023-12-19", 'minute', False, True)

# print(histData)

# print(type(histData))

# Convert to DataFrame
df = pd.DataFrame(histData)

# Feature selection - you might want to select relevant features
features = ['open', 'high', 'low', 'volume']

# Scaling features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[features])


# Function to create sequences
def create_dataset(data, time_step=1, future_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - future_step):
        X.append(data[i:(i + time_step), :])  # All features
        y.append(data[i + time_step + future_step - 1, 0])  # Predicting the 'close' price at future_step
    return np.array(X), np.array(y)


# Splitting dataset into train and test split
# Define time steps and future steps
time_step = 100
future_step_10 = 10  # 10 minutes into the future
future_step_30 = 30  # 30 minutes into the future
future_step_60 = 60  # 60 minutes into the future


# Create datasets for each future step
X_10, y_10 = create_dataset(scaled_data, time_step, future_step_10)
X_30, y_30 = create_dataset(scaled_data, time_step, future_step_30)
X_60, y_60 = create_dataset(scaled_data, time_step, future_step_60)


# Example: Train and predict for 10th minute
# Splitting dataset into train and test split for 10th minute
X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(X_10, y_10, test_size=0.2, random_state=42)

# Reshape input for LSTM
X_train_10 = X_train_10.reshape(X_train_10.shape[0], X_train_10.shape[1], len(features))
X_test_10 = X_test_10.reshape(X_test_10.shape[0], X_test_10.shape[1], len(features))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, len(features))))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# print("x train shape : ")
# print(X_train.shape)
# print("y train shape : ")
# print(y_train.shape)

# Train the model
model.fit(X_train_10, y_train_10, batch_size=1, epochs=1)

# Predicting the stock price for the 10th minute
# You need to select the appropriate time window from your data
last_time_step_data = scaled_data[-(time_step+future_step_10):]
prediction_input = last_time_step_data[:time_step]
prediction_input = prediction_input.reshape(1, time_step, len(features))
predicted_price_10 = model.predict(prediction_input)

# Note: This part of the code is very simplified and should be adjusted for your specific needs
# last_60_minutes = scaled_data[-time_step:]
# last_60_minutes = last_60_minutes.reshape(1, time_step, len(features))
# predicted_price = model.predict(last_60_minutes)
# predicted_price = scaler.inverse_transform(predicted_price)  # Inverse transform to get actual value

# print("Predicted Close Price for next 60 minutes:", predicted_price[0][0])

# Assuming 'predicted_price' has the shape (1, 1)
# Create a dummy array with the same number of features
dummy_array = np.zeros((predicted_price_10.shape[0], scaled_data.shape[1]))

# Place the predicted price in the first column (or the relevant column for 'close' price)
dummy_array[:, 0] = predicted_price_10.ravel()

# Apply inverse transform
inverse_transformed = scaler.inverse_transform(dummy_array)

# Extract the relevant predicted price
final_predicted_price = inverse_transformed[:, 0]  # Assuming 'close' price is the first feature

# Get the current time
current_time = datetime.now()

# Calculate the time for the 10th minute from now
time_10th_minute_from_now = current_time + timedelta(minutes=10)

print("Predicted Close Price for 10th minute:", final_predicted_price[0])

# Print the calculated time
print("Time for the 10th minute from now:", time_10th_minute_from_now.strftime('%Y-%m-%d %H:%M:%S'))  # 1207.50 (actual val)   1241.53 (pred val)
