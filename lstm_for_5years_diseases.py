import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# =====================================================
# 1. Load Training Data (Historical Disease Dataset)
# =====================================================
data = pd.read_csv("Rice_Deseases_Factors_550.csv")

# Normalize column names
data.columns = data.columns.str.strip().str.replace("Â", "").str.replace(" ", "")

# Normalize column names
data.columns = data.columns.str.strip().str.lower()

# Map possible variations → standard names
train_map = {
    'temperature (°c)': "Temperature (°C)",
    'temperature (c)': "Temperature (°C)",
    'temperature': "Temperature (°C)",
    'temp': "Temperature (°C)",

    'humidity (%)': "Humidity (%)",
    'humidity': "Humidity (%)",
    'relative humidity': "Humidity (%)",

    'rainfall (mm)': "Rainfall (mm)",
    'rainfall': "Rainfall (mm)",

    'soil moisture (%)': "Soil Moisture (%)",
    'soil moisture': "Soil Moisture (%)",

    'ph level': "pH Level",
    'ph': "pH Level",

    'nitrogen content (mg/kg)': "Nitrogen Content (mg/kg)",
    'nitrogen': "Nitrogen Content (mg/kg)",

    'potassium content (mg/kg)': "Potassium Content (mg/kg)",
    'potassium': "Potassium Content (mg/kg)",

    'wind speed (m/s)': "Wind Speed (m/s)",
    'wind speed': "Wind Speed (m/s)",

    'disease presence': "Disease Presence",
    'presence': "Disease Presence"
}

# Apply mapping
data = data.rename(columns={k: v for k, v in train_map.items() if k in data.columns})

# Expected features
features = [
    "Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "Soil Moisture (%)",
    "pH Level", "Nitrogen Content (mg/kg)", "Potassium Content (mg/kg)", "Wind Speed (m/s)"
]
target = "Disease Presence"

# Check again after renaming
for col in features + [target]:
    if col not in data.columns:
        raise ValueError(f"Still missing column in training data: {col}")



# Ensure all expected columns exist
for col in features + [target]:
    if col not in data.columns:
        raise ValueError(f"Missing column in training data: {col}")

# Clean numeric columns
for col in features:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing with median
data = data.fillna(data.median(numeric_only=True))

X = data[features].values
y = data[target].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# 2. Prepare Sequential Data for LSTM
# =====================================================
def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 30
X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

# Train/test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# =====================================================
# 3. Build & Train LSTM Model
# =====================================================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# =====================================================
# 4. Evaluate Short-Term Forecast (60 days)
# =====================================================
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nLSTM Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# =====================================================
# 5. Load 5-Year Forecast Data (Climate Scenarios)
# =====================================================
forecast = pd.read_csv("rice_forecast_5years.csv")
forecast.columns = forecast.columns.str.strip().str.lower()

# Map possible column names → training feature names
forecast_map = {
    'average temperature': "Temperature (°C)",
    'maximum temperature': "Temperature (°C)",
    'minimum temperature': "Temperature (°C)",
    'temperature': "Temperature (°C)",
    'average relative humidity': "Humidity (%)",
    'humidity': "Humidity (%)",
    'total rainfall': "Rainfall (mm)",
    'rainfall': "Rainfall (mm)",
    'soil moisture': "Soil Moisture (%)",
    'average ph': "pH Level",
    'ph': "pH Level",
    'nitrogen': "Nitrogen Content (mg/kg)",
    'nitrogen content': "Nitrogen Content (mg/kg)",
    'potassium': "Potassium Content (mg/kg)",
    'potassium content': "Potassium Content (mg/kg)",
    'average wind speed': "Wind Speed (m/s)",
    'wind speed': "Wind Speed (m/s)",
    'date': "Date"
}

forecast = forecast.rename(columns={k: v for k, v in forecast_map.items() if k in forecast.columns})

# Build forecast features
future_features = pd.DataFrame()
for col in features:
    if col in forecast.columns:
        future_features[col] = pd.to_numeric(forecast[col], errors="coerce")
    else:
        print(f"⚠️ Warning: '{col}' missing in forecast file → filling with training median")
        future_features[col] = data[col].median()

# Fill NaN
future_features = future_features.fillna(data[features].median())

# Scale forecast features
future_scaled = scaler.transform(future_features.values)

# Add dates
if "Date" in forecast.columns:
    forecast["Date"] = pd.to_datetime(forecast["Date"], errors="coerce")
else:
    forecast["Date"] = pd.date_range(start="2025-01-01", periods=len(forecast))

# =====================================================
# 6. Forecast Future Disease Risk (5 years)
# =====================================================
X_future, _ = create_sequences(future_scaled, np.zeros(len(future_scaled)), time_steps)
future_preds = (model.predict(X_future) > 0.5).astype("int32")

# Align with dates
forecast = forecast.iloc[time_steps:].copy()
forecast["Predicted Disease Risk"] = future_preds

# Save forecast results
forecast.to_csv("5year_disease_forecast.csv", index=False)
print("\n✅ 5-Year Forecast saved to 5year_disease_forecast.csv")

# =====================================================
# 7. Visualization
# =====================================================
plt.figure(figsize=(12, 6))
plt.plot(forecast["Date"], forecast["Predicted Disease Risk"], label="Predicted Disease Risk")
plt.title("5-Year Rice Disease Forecast (LSTM)")
plt.xlabel("Date")
plt.ylabel("Risk (0=Low, 1=High)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("5year_disease_forecast.png")
plt.show()
