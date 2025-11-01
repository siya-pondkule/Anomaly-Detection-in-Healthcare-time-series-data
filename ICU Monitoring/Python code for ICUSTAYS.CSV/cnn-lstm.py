import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, RepeatVector, TimeDistributed, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\CNN-LSTM-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Load and preprocess data
# ======================
print(f"\n=== Processing {DATA_PATH.name} for CNN-LSTM Autoencoder ===")
df = pd.read_csv(DATA_PATH)

# Select only numeric columns (remove IDs/timestamps)
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for CNN-LSTM anomaly detection.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================
# Create sequences
# ======================
time_steps = 5  # length of time window

def create_sequences(data, time_steps=5):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

X_seq = create_sequences(X_scaled, time_steps)
print(f"Shape of data for CNN-LSTM: {X_seq.shape}")  # (samples, timesteps, features)

# ======================
# Build CNN-LSTM Autoencoder
# ======================
def build_cnn_lstm_autoencoder(timesteps, n_features):
    model = Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(timesteps, n_features)),
        MaxPooling1D(pool_size=2),
        LSTM(64, activation='relu', return_sequences=False),
        RepeatVector(timesteps - 1),  # adjust to match reduced sequence after pooling
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_cnn_lstm_autoencoder(X_seq.shape[1], X_seq.shape[2])
model.summary()

# ======================
# Train model
# ======================
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\nTraining CNN-LSTM Autoencoder...")
history = model.fit(
    X_seq, X_seq[:, 1:, :],  # since pooling reduces timesteps by 1
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop]
)

# ======================
# Reconstruction and anomaly detection
# ======================
X_pred = model.predict(X_seq)
mse = np.mean(np.mean(np.square(X_seq[:, 1:, :] - X_pred), axis=2), axis=1)  # sequence-wise MSE

# Use percentile-based threshold
threshold = np.percentile(mse, 95)
anomalies = np.where(mse > threshold)[0]

print(f"\nDetected {len(anomalies)} anomalies out of {len(mse)} sequences.")
print(f"Anomaly threshold: {threshold:.6f}")

# ======================
# Save results
# ======================
results_df = pd.DataFrame({
    "Sequence_Index": np.arange(len(mse)),
    "Reconstruction_Error": mse,
    "Anomaly_Label": np.where(mse > threshold, "Anomaly", "Normal")
})
results_file = OUTPUT_DIR / "ICUSTAYS_CNNLSTM_Results.csv"
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
summary_file = SUMMARY_DIR / "CNNLSTM_ICUSTAYS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved → {summary_file}")

# ======================
# Plot reconstruction error
# ======================
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error", color="blue", alpha=0.7)
plt.scatter(anomalies, mse[anomalies], color="red", label="Detected Anomalies", marker="x")
plt.axhline(threshold, color="orange", linestyle="--", label="Threshold")
plt.title("CNN-LSTM Autoencoder Anomaly Detection - ICUSTAYS.csv")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_CNNLSTM_Anomaly_Plot.png")
plt.close()

print(f"Plot saved → {OUTPUT_DIR / 'ICUSTAYS_CNNLSTM_Anomaly_Plot.png'}")

print("\n✅ CNN-LSTM Autoencoder anomaly detection complete!")
