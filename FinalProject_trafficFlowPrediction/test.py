""" import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
data = pd.read_csv('Data/traffic.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data = data.sort_values('DateTime')

# Extract relevant columns
traffic_data = data['Vehicles'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
traffic_data_scaled = scaler.fit_transform(traffic_data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 24  # Using the past 24 hours to predict the next hour
X, y = create_sequences(traffic_data_scaled, seq_length)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)

# Reverse normalization
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Evaluate performance
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}") """


import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Data/traffic.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data = data.sort_values('DateTime')
data.set_index('DateTime', inplace=True)  # Set DateTime as index
traffic_data = data['Vehicles']

""" # Plot the data
traffic_data.plot(figsize=(10, 6))
plt.title("Traffic Flow Data")
plt.show() """

# Split the data into train and test sets
train_size = int(len(traffic_data) * 0.8)
train, test = traffic_data[:train_size], traffic_data[train_size:]

# Define the SARIMA model
sarima_model = SARIMAX(train, 
                       order=(1, 1, 1),  # ARIMA parameters (p, d, q)
                       seasonal_order=(1, 1, 1, 24),  # Seasonal parameters (P, D, Q, s)
                       enforce_stationarity=False, 
                       enforce_invertibility=False)

# Fit the model
sarima_result = sarima_model.fit(disp=False)

# Print the summary
print(sarima_result.summary())

# Forecast
forecast = sarima_result.forecast(steps=len(test))

# Evaluate the model
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

# Plot actual vs forecast
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual Traffic')
plt.plot(test.index, forecast, label='Forecasted Traffic')
plt.legend()
plt.title("Actual vs Forecasted Traffic Flow")
plt.show()