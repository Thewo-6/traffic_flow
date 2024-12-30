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