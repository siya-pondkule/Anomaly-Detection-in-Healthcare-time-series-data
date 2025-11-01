import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\LSTM-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)
# ======================
# Load and preprocess data
# ======================
print(f"\n=== Processing {DATA_PATH.name} for LSTM Autoencoder ===")
df = pd.read_csv(DATA_PATH)

# Select only numeric and relevant columns
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for LSTM Autoencoder anomaly detection.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# Reshape for LSTM [samples, timesteps, features]
# Here we treat each ICU stay as part of a sequential window
time_steps = 5  # number of records per sequence
def create_sequences(data, time_steps=5):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

X_seq = create_sequences(X_scaled, time_steps)
print(f"Shape of data for LSTM: {X_seq.shape}")  # (samples, timesteps, features)

# ======================
# LSTM Autoencoder Model
# ======================
def build_lstm_autoencoder(timesteps, n_features):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(timesteps, n_features), return_sequences=False),
        RepeatVector(timesteps),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_lstm_autoencoder(X_seq.shape[1], X_seq.shape[2])

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\nTraining LSTM Autoencoder...")
history = model.fit(
    X_seq, X_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop]
)

# ======================
# Reconstruction and Anomaly Detection
# ======================
X_pred = model.predict(X_seq)
mse = np.mean(np.mean(np.square(X_seq - X_pred), axis=2), axis=1)  # per sequence error

# Use percentile-based threshold
threshold = np.percentile(mse, 95)
anomalies = np.where(mse > threshold)[0]

print(f"\nDetected {len(anomalies)} anomalies out of {len(mse)} sequences.")
print(f"Threshold for anomaly detection: {threshold:.6f}")

# ======================
# Save Results
# ======================
results_df = pd.DataFrame({
    "Sequence_Index": np.arange(len(mse)),
    "Reconstruction_Error": mse,
    "Anomaly_Label": np.where(mse > threshold, "Anomaly", "Normal")
})
results_file = OUTPUT_DIR / "ICUSTAYS_LSTM_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# ======================
# Summary
# ======================
summary_data = {
    "Total Sequences": len(mse),
    "Detected Anomalies": len(anomalies),
    "Anomaly Percentage (%)": round(100 * len(anomalies) / len(mse), 2)
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "LSTM_ICUSTAYS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved → {summary_file}")

# ======================
# Plot Reconstruction Error
# ======================
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error", color="blue", alpha=0.7)
plt.scatter(anomalies, mse[anomalies], color="red", label="Detected Anomalies", marker="x")
plt.axhline(threshold, color="orange", linestyle="--", label="Anomaly Threshold")
plt.title("LSTM Autoencoder Anomaly Detection - ICUSTAYS.csv")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_LSTM_Anomaly_Plot.png")
plt.close()

print(f"Plot saved → {OUTPUT_DIR / 'ICUSTAYS_LSTM_Anomaly_Plot.png'}")

print("\n✅ LSTM Autoencoder anomaly detection complete!")
