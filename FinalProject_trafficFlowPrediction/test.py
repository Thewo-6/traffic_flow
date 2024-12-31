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


""" import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Data/traffic.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data = data.sort_values('DateTime')
data.set_index('DateTime', inplace=True)  # Set DateTime as index
traffic_data = data['Vehicles']


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
plt.show() """

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Loading the data
traffic_data = pd.read_csv('Data/traffic.csv')

# Convert DateTime to pandas datetime object
traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'])

# Extract time-based features
traffic_data['Hour'] = traffic_data['DateTime'].dt.hour
traffic_data['DayOfWeek'] = traffic_data['DateTime'].dt.dayofweek
traffic_data['Month'] = traffic_data['DateTime'].dt.month

# One-hot encode the 'Junction' column
encoder = OneHotEncoder(sparse_output=False, drop='first')
junction_encoded = encoder.fit_transform(traffic_data[['Junction']])
junction_columns = [f"Junction_{int(i)}" for i in encoder.categories_[0][1:]]
junction_df = pd.DataFrame(junction_encoded, columns=junction_columns)

# Combine the processed features with the original data
processed_data = pd.concat([traffic_data, junction_df], axis=1)

# Drop unnecessary columns
processed_data.drop(columns=['DateTime', 'ID', 'Junction'], inplace=True)

# Split into features (X) and target (y)
X = processed_data.drop(columns=['Vehicles'])
y = processed_data['Vehicles']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Display the first few rows of the processed training data
print(pd.DataFrame(X_train, columns=X.columns).head())

#processed_data.to_csv("precessed_data.csv", index=False)

# Initialize the model
#model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model the training data
#model.fit(X_train, y_train)

# Train the model
rf_model.fit(X_train, y_train)

# Function to predict the number of vehicles for a specific input
def predict_traffic(hour, day_of_week, month, junction):
    # Prepare the input data
    input_data = {
        'Hour': [hour],
        'DayOfWeek': [day_of_week],
        'Month': [month]
    }
    # Add one-hot encoded junction
    for col in junction_columns:
        input_data[col] = [1 if col == f"Junction_{junction}" else 0]
    
    # Convert to DataFrame and scale
    input_df = pd.DataFrame(input_data)
    input_scaled = scaler.transform(input_df)
    
    # Predict using the trained model
    predicted_vehicles = rf_model.predict(input_scaled)
    return int(round(predicted_vehicles[0]))
# Take inputs from the user
print("Enter the details for traffic prediction:")

hour = int(input("Hour (0-23): "))  # Ask the user to input the hour
day_of_week = int(input("Day of Week (0 = Monday, 6 = Sunday): "))  # Ask for the day of the week
month = int(input("Month (1-12): "))  # Ask for the month
junction = int(input("Junction (1, 2, 3, or 4): "))  # Ask for the junction ID

# Predict traffic based on the inputs
predicted_traffic = predict_traffic(hour, day_of_week, month, junction)

# Display the result
print(f"\nPredicted Traffic at Hour {hour}, Day {day_of_week}, Month {month}, Junction {junction}: {predicted_traffic} vehicles")

# Predict on the test data
y_pred_rf = rf_model.predict(X_test)

# Calculate evaluation metrics
#mae = mean_absolute_error(y_test, y_pred)
#mse = mean_squared_error(y_test, y_pred)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print(f"Mean Absolute Error (MAE): {mae_rf:.2f}")
print(f"Mean Squared Error (MSE): {mse_rf:.2f}")

features = ['Hour', 'DayOfWeek', 'Month', 'Junction_2', 'Junction_3', 'Junction_4']

feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importances)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importances for Traffic Prediction', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.show()

print("\n")

# Actual vs Predicted Traffic Flow
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:50], label='Actual Traffic Flow', marker='o', linestyle='dashed', alpha=0.7)
plt.plot(y_pred_rf[:50], label='Predicted Traffic Flow', marker='o', alpha=0.7)
plt.title('Actual vs Predicted Traffic Flow (First 50 Samples)', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Number of Vehicles', fontsize=12)
plt.legend()
plt.show()

print("\n")

# Error distribution
errors = y_test - y_pred_rf
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, color='red', bins=30, alpha=0.7)
plt.title('Distribution of Prediction Errors', fontsize=16)
plt.xlabel('Error (Actual - Predicted)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

print("\n")

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(processed_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.show()