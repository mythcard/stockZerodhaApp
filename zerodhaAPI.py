#!python
import logging
from kiteconnect import KiteConnect

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
# data = kite.instruments()

histData = kite.historical_data('408065', '2023-12-11', "2023-12-14", 'minute', False, True)

print(histData)

# print(type(histData))

# Convert to DataFrame
df = pd.DataFrame(histData)

# Feature selection - you might want to select relevant features
features = ['open', 'high', 'low', 'volume']

# Scaling features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[features])


# Function to create sequences
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])  # All features
        y.append(data[i + time_step, 0])  # Predicting the 'close' price
    return np.array(X), np.array(y)


# Splitting dataset into train and test split
time_step = 100
X, y = create_dataset(scaled_data, time_step)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(features))
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], len(features))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, len(features))))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
print("x train shape : ")
print(X_train.shape)
print("y train shape : ")
print(y_train.shape)

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Predicting next 60 minutes
# Note: This part of the code is very simplified and should be adjusted for your specific needs
last_60_minutes = scaled_data[-time_step:]
last_60_minutes = last_60_minutes.reshape(1, time_step, len(features))
predicted_price = model.predict(last_60_minutes)
# predicted_price = scaler.inverse_transform(predicted_price)  # Inverse transform to get actual value

# print("Predicted Close Price for next 60 minutes:", predicted_price[0][0])

# Assuming 'predicted_price' has the shape (1, 1)
# Create a dummy array with the same number of features
dummy_array = np.zeros((predicted_price.shape[0], scaled_data.shape[1]))

# Place the predicted price in the first column (or the relevant column for 'close' price)
dummy_array[:, 0] = predicted_price.ravel()

# Apply inverse transform
inverse_transformed = scaler.inverse_transform(dummy_array)

# Extract the relevant predicted price
final_predicted_price = inverse_transformed[:, 0]  # Assuming 'close' price is the first feature

print("Predicted Close Price for next minute:", final_predicted_price[0])
