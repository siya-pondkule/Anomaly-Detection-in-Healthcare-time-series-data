import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, UpSampling1D, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# ====================== PATHS ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\CNN-LSTM-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for CNN-LSTM Anomaly Detection ===")

# ====================== Detect Encoding & Delimiter ======================
try:
    with open(DATA_PATH, "rb") as f:
        sample = f.read(4096)
    encoding = "utf-8"
    sample.decode(encoding)
except Exception:
    encoding = "latin1"
print(f"Detected encoding: {encoding}")

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    lines = [next(f) for _ in range(10)]
sample_text = "\n".join(lines)
delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: sample_text.count(d))
print(f"Detected delimiter: '{best_delim}'")

# ====================== Load Data ======================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ Loaded data → shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ====================== Preprocessing ======================
# Convert timestamps
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Create time difference feature
if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.append("time_diff_min")

df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna(how="any")
if df_numeric.empty:
    raise ValueError("❌ No valid numeric data for CNN-LSTM training.")
print(f"🧮 Using numeric columns: {numeric_cols}")

# ====================== Normalize ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ====================== Create Sequences ======================
TIME_STEPS = 10

def create_sequences(X, time_steps=TIME_STEPS):
    seqs = []
    for i in range(len(X) - time_steps):
        seqs.append(X[i:(i + time_steps)])
    return np.array(seqs)

X_seq = create_sequences(X_scaled)
print(f"✅ Created sequences → shape: {X_seq.shape}")

# ====================== Build CNN-LSTM Autoencoder ======================
# ====================== Build CNN-LSTM Autoencoder ======================
model = Sequential([
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2])),
    MaxPooling1D(pool_size=2, padding='same'),
    LSTM(64, activation='relu', return_sequences=False),
    RepeatVector(X_seq.shape[1]),  # repeats 10 timesteps
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_seq.shape[2]))  # output shape: (None, 10, features)
])

model.compile(optimizer='adam', loss='mse')
model.summary()


# ====================== Train Model ======================
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_seq, X_seq,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ====================== Compute Reconstruction Error ======================
X_pred = model.predict(X_seq)
recon_error = np.mean(np.square(X_seq - X_pred), axis=(1, 2))

threshold = np.percentile(recon_error, 95)
anomalies = recon_error > threshold

print(f"✅ Training complete.")
print(f"🚨 Detected {anomalies.sum()} anomalies out of {len(anomalies)} sequences.")

# ====================== Save Results ======================
results_df = pd.DataFrame({
    "sequence_index": np.arange(len(recon_error)),
    "reconstruction_error": recon_error,
    "is_anomaly": anomalies.astype(int)
})
results_file = OUTPUT_DIR / "LABEVENTS_CNN_LSTM_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"✅ Results saved → {results_file}")

summary = {
    "total_sequences": len(anomalies),
    "detected_anomalies": int(anomalies.sum()),
    "anomaly_percentage": round(100.0 * anomalies.sum() / len(anomalies), 3),
    "threshold": round(threshold, 5)
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "CNN_LSTM_LABEVENTS_Summary.csv", index=False)
print(f"✅ Summary saved → {SUMMARY_DIR / 'CNN_LSTM_LABEVENTS_Summary.csv'}")

# ====================== Visualization ======================
plt.figure(figsize=(12,6))
plt.title("CNN-LSTM Autoencoder Reconstruction Error - LABEVENTS")
plt.plot(recon_error, label="Reconstruction Error", alpha=0.7)
plt.axhline(threshold, color="red", linestyle="--", label="Anomaly Threshold")
plt.scatter(np.where(anomalies)[0], recon_error[anomalies], color="red", marker="x", label="Anomalies")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_CNN_LSTM_AnomalyPlot.png")
plt.close()
print(f"✅ Plot saved → {OUTPUT_DIR / 'LABEVENTS_CNN_LSTM_AnomalyPlot.png'}")

print("\n🎯 CNN-LSTM Anomaly Detection completed successfully!")
