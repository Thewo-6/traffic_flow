from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

# Convert DateTime to pandas datetime object
traffic_data['DateTime'] = pd.to_datetime(traffic_data['DateTime'])

# Extract time-based features
traffic_data['Hour'] = traffic_data['DateTime'].dt.hour
traffic_data['DayOfWeek'] = traffic_data['DateTime'].dt.dayofweek
traffic_data['Month'] = traffic_data['DateTime'].dt.month

# One-hot encode the 'Junction' column
encoder = OneHotEncoder(sparse=False, drop='first')
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
pd.DataFrame(X_train, columns=X.columns).head()