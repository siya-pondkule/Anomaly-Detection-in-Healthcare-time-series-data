import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# ====================== PATHS ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\LSTM-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for LSTM Autoencoder Anomaly Detection ===")

# ====================== Detect Encoding & Delimiter ======================
try:
    with open(DATA_PATH, "rb") as f:
        sample = f.read(4096)
    encoding = "utf-8"
    sample.decode(encoding)
except Exception:
    encoding = "latin1"
print(f"Detected encoding: {encoding}")

# Detect delimiter
with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    lines = [next(f) for _ in range(10)]
sample_text = "\n".join(lines)
delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: sample_text.count(d))
print(f"Detected delimiter: '{best_delim}'")

# ====================== Load CSV ======================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ Loaded data → shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ====================== Preprocessing ======================
# Convert time columns
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Create time-based feature
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
    raise ValueError("❌ No valid numeric data for LSTM training.")
print(f"🧮 Using numeric columns: {numeric_cols}")

# ====================== Normalize ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ====================== Prepare Data for LSTM ======================
TIME_STEPS = 10  # Number of time steps per sequence

def create_sequences(X, time_steps=TIME_STEPS):
    sequences = []
    for i in range(len(X) - time_steps):
        sequences.append(X[i:(i + time_steps)])
    return np.array(sequences)

X_seq = create_sequences(X_scaled, TIME_STEPS)
print(f"✅ Created sequences → shape: {X_seq.shape}")

# ====================== Build LSTM Autoencoder ======================
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    RepeatVector(X_seq.shape[1]),
    LSTM(32, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_seq.shape[2]))
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

# ====================== Reconstruction Error ======================
X_pred = model.predict(X_seq)
recon_error = np.mean(np.square(X_seq - X_pred), axis=(1, 2))

threshold = np.percentile(recon_error, 95)  # Top 5% as anomalies
anomalies = recon_error > threshold

print(f"✅ Training complete.")
print(f"🚨 Detected {anomalies.sum()} anomalies out of {len(anomalies)} sequences.")

# ====================== Save Results ======================
results_df = pd.DataFrame({
    "sequence_index": np.arange(len(recon_error)),
    "reconstruction_error": recon_error,
    "is_anomaly": anomalies.astype(int)
})
results_file = OUTPUT_DIR / "LABEVENTS_LSTM_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"✅ Results saved → {results_file}")

summary = {
    "total_sequences": len(anomalies),
    "detected_anomalies": int(anomalies.sum()),
    "anomaly_percentage": round(100.0 * anomalies.sum() / len(anomalies), 3),
    "threshold": round(threshold, 5)
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "LSTM_LABEVENTS_Summary.csv", index=False)
print(f"✅ Summary saved → {SUMMARY_DIR / 'LSTM_LABEVENTS_Summary.csv'}")

# ====================== Plot Results ======================
plt.figure(figsize=(12,6))
plt.title("LSTM Autoencoder Reconstruction Error - LABEVENTS")
plt.plot(recon_error, label="Reconstruction Error", alpha=0.7)
plt.axhline(threshold, color="red", linestyle="--", label="Anomaly Threshold")
plt.scatter(np.where(anomalies)[0], recon_error[anomalies], color="red", marker="x", label="Anomalies")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_LSTM_AnomalyPlot.png")
plt.close()
print(f"✅ Plot saved → {OUTPUT_DIR / 'LABEVENTS_LSTM_AnomalyPlot.png'}")

print("\n🎯 LSTM Autoencoder training and anomaly detection completed successfully!")
