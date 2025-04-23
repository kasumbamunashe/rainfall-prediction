# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV  # Added GridSearchCV here
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

# Load the dataset
data = pd.read_csv("rainfall_dataset_updated.csv")

# Rename columns for consistency
data.rename(columns={
    'Annual Mean Precipit': 'Rainfall',
    '5-yr smooth Precipit': 'Rainfall_Smooth',
    'Annual Mean Temp': 'Temperature',
    '5-yr smooth Temp': 'Temperature_Smooth',
    'Humidity (%)': 'Humidity',
    'Wind Speed (km/h)': 'Wind_Speed',
    'Air Pressure (hPa)': 'Pressure',
    'Dew Point (°C)': 'Dew_Point',
    'Solar Radiation (W/m²)': 'Solar_Radiation',
    'Cloud Cover (%)': 'Cloud_Cover',
    'Elevation (m)': 'Elevation'
}, inplace=True)

# Create lag features
lags = [1, 2]
for lag in lags:
    data[f'Rainfall_Lag{lag}'] = data['Rainfall'].shift(lag)
    data[f'Temperature_Lag{lag}'] = data['Temperature'].shift(lag)

data['Dew_Point_Lag1'] = data['Dew_Point'].shift(1)

# Drop NaN values caused by lag features
data.dropna(inplace=True)

# Select important features
features = [
    'Temperature', 'Dew_Point', 'Humidity', 'Wind_Speed', 'Pressure',
    'Solar_Radiation', 'Cloud_Cover', 'Elevation', 'Rainfall_Lag1',
    'Rainfall_Lag2', 'Temperature_Lag1', 'Temperature_Lag2', 'Dew_Point_Lag1'
]

X = data[features]
y = data['Rainfall']

# Feature Scaling: Standardize features to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset (time-series split for cross-validation)
# Use TimeSeriesSplit for time-series data to avoid data leakage
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Initialize the models with early stopping
model = XGBRegressor(random_state=42, early_stopping_rounds=10, eval_metric='rmse')

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Perform GridSearchCV with TimeSeriesSplit
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Get the best models
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate models
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("Best Parameters:", grid_search.best_params_)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the trained models and scaler
joblib.dump(best_model, "rainfall_model_final.pkl")
joblib.dump(scaler, "scaler.pkl")  # Save the scaler for future use