import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# 1. Load 5-Year Forecast Dataset
# =====================================================
data = pd.read_csv("rice_forecast_5years.csv")

# Normalize column names
data.columns = data.columns.str.strip().str.lower()

# Ensure standard names
feature_map = {
    'temperature': 'Temperature',
    'humidity': 'Humidity',
    'rainfall': 'Rainfall',
    'soil moisture': 'Soil Moisture',
    'ph level': 'pH Level',
    'nitrogen content': 'Nitrogen Content',
    'potassium content': 'Potassium Content',
    'wind speed': 'Wind Speed',
    'disease risk': 'Disease Risk',
    'disease type': 'Disease Type'
}
data = data.rename(columns={k: v for k, v in feature_map.items() if k in data.columns})

# Ensure Date column
if 'date' in data.columns:
    data['Date'] = pd.to_datetime(data['date'], errors='coerce')

# Features and target
features = ['Temperature','Humidity','Rainfall','Soil Moisture','pH Level',
            'Nitrogen Content','Potassium Content','Wind Speed']
target = 'Disease Risk'

# Clean numeric
for col in features + [target]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data[features] = data[features].fillna(data[features].median())

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data[features].values)
y = data[target].values

# =====================================================
# 2. Sequence Preparation
# =====================================================
def create_sequences(X, y, seq_len=30):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i+seq_len)])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

SEQ_LEN = 30
X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)

split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# =====================================================
# 3. Build & Train LSTM
# =====================================================
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, X_seq.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50, batch_size=16, callbacks=[es], verbose=1)

# =====================================================
# 4. Accuracy Evaluation
# =====================================================
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

short_term_acc = accuracy_score(y_test[:60], y_pred[:60]) if len(y_test) >= 60 else None
long_term_acc = accuracy_score(y_test, y_pred)

print(f"\nShort-term (60 days) Accuracy: {short_term_acc:.3f}" if short_term_acc else "Not enough data for 60-day accuracy")
print(f"Long-term (5 years) Accuracy: {long_term_acc:.3f}")

# Seasonal-naive baseline
if len(y_test) > 365:
    naive_pred = np.roll(y_test, 365)[:len(y_test)]
    naive_acc = accuracy_score(y_test[365:], naive_pred[365:])
    print(f"Seasonal-Naive Accuracy: {naive_acc:.3f}")

# =====================================================
# 5. Planting Windows
# =====================================================
data['Season'] = data['Date'].dt.month.apply(lambda m: 'Maha' if m in [10,11,12,1,2,3] else 'Yala')
seasonal_risk = data.groupby('Season')[target].mean()
print("\nAverage Risk by Season:\n", seasonal_risk)

# =====================================================
# 6. Climate Scenarios
# =====================================================
scenarios = {
    'Baseline': data[features],
    'Temp+2C': data[features].assign(Temperature=data['Temperature']+2),
    'Rain+10%': data[features].assign(Rainfall=data['Rainfall']*1.1),
    'Rain-10%': data[features].assign(Rainfall=data['Rainfall']*0.9),
}
scenario_results = {}
for name, df in scenarios.items():
    scaled = scaler.transform(df)
    seqs = []
    cur = X_scaled[-SEQ_LEN:]
    for i in range(len(scaled)):
        cur = np.vstack([cur[1:], scaled[i]])
        seqs.append(cur)
    preds = (model.predict(np.array(seqs)) > 0.5).astype(int).flatten()
    scenario_results[name] = preds

# =====================================================
# 7. Visualizations (Line Graphs)
# =====================================================
os.makedirs("output", exist_ok=True)

# Training Loss
plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Training Loss"); plt.savefig("output/train_loss.png"); plt.show()

# Forecast Risk (full horizon)
plt.figure(figsize=(14,6))
plt.plot(data['Date'], data['Disease Risk'], label="Observed Risk")
plt.plot(data['Date'][SEQ_LEN:], y_pred, label="LSTM Predicted")
plt.legend(); plt.title("Disease Risk Forecast (5 Years)")
plt.savefig("output/forecast_risk.png"); plt.show()

# Short-term vs Long-term accuracy visual
plt.figure(figsize=(14,6))
plt.plot(data['Date'][SEQ_LEN:SEQ_LEN+60], y_pred[:60], label="Short-term Forecast")
plt.plot(data['Date'][SEQ_LEN:], y_pred, alpha=0.3, label="Long-term Forecast")
plt.legend(); plt.title("Short-term (60 days) vs Long-term (5 years) Forecast")
plt.savefig("output/short_vs_long.png"); plt.show()

# Climate scenarios
plt.figure(figsize=(14,6))
for name, preds in scenario_results.items():
    plt.plot(data['Date'], preds, label=name)
plt.legend(); plt.title("Climate Change Scenario Comparisons")
plt.savefig("output/climate_scenarios.png"); plt.show()

# Seasonal averages
sns.lineplot(data=data, x="Date", y="Disease Risk", hue="Season")
plt.title("Seasonal Disease Risk Patterns (Maha vs Yala)")
plt.savefig("output/seasonal_patterns.png"); plt.show()

# Disease-type risk patterns
if 'Disease Type' in data.columns:
    plt.figure(figsize=(14,6))
    sns.lineplot(data=data, x="Date", y="Disease Risk", hue="Disease Type")
    plt.title("Disease Risk Patterns by Type")
    plt.savefig("output/disease_type_patterns.png")
    plt.show()
