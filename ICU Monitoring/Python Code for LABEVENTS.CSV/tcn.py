import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ===================== PATHS =====================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\TCN-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for TCN-based Anomaly Detection ===")

# ===================== ENCODING & DELIMITER =====================
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

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ Loaded data → shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ===================== PREPROCESS =====================
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
    raise ValueError("❌ No valid numeric data for TCN training.")
print(f"🧮 Using numeric columns: {numeric_cols}")

# ===================== NORMALIZE =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ===================== CREATE SEQUENCES =====================
TIME_STEPS = 20

def create_sequences(X, time_steps=TIME_STEPS):
    seqs = []
    for i in range(len(X) - time_steps):
        seqs.append(X[i:(i + time_steps)])
    return np.array(seqs)

X_seq = create_sequences(X_scaled)
print(f"✅ Created sequences → shape: {X_seq.shape}")

if len(X_seq) < 10:
    raise ValueError("Not enough sequences to train TCN. Try reducing TIME_STEPS.")

# ===================== BUILD TCN AUTOENCODER =====================
input_shape = (X_seq.shape[1], X_seq.shape[2])
inputs = Input(shape=input_shape)

# Encoder: Temporal Conv layers with dilation
x = Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu', dilation_rate=1)(inputs)
x = Dropout(0.2)(x)
x = Conv1D(filters=128, kernel_size=3, padding='causal', activation='relu', dilation_rate=2)(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
encoded = Dense(64, activation='relu')(x)

# Decoder: Mirror the encoder
x = Dense(X_seq.shape[1] * X_seq.shape[2], activation='relu')(encoded)
x = Reshape((X_seq.shape[1], X_seq.shape[2]))(x)
x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
x = Dropout(0.2)(x)
x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
decoded = Dense(X_seq.shape[2], activation='linear')(x)

model = Model(inputs, decoded)
model.compile(optimizer=Adam(1e-3), loss='mse')
model.summary()

# ===================== TRAIN MODEL =====================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_seq, X_seq,
    epochs=40,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ===================== RECONSTRUCTION ERROR =====================
X_pred = model.predict(X_seq)
mse_seq = np.mean(np.mean(np.square(X_seq - X_pred), axis=2), axis=1)
threshold = np.percentile(mse_seq, 95)
anomalies = mse_seq > threshold

print(f"🚨 Detected {anomalies.sum()} anomalies out of {len(mse_seq)} sequences (threshold={threshold:.6f})")

# ===================== SAVE RESULTS =====================
results_df = pd.DataFrame({
    "sequence_index": np.arange(len(mse_seq)),
    "reconstruction_error": mse_seq,
    "is_anomaly": anomalies.astype(int)
})
results_file = OUTPUT_DIR / "LABEVENTS_TCN_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"✅ Results saved → {results_file}")

summary = {
    "total_sequences": len(mse_seq),
    "detected_anomalies": int(anomalies.sum()),
    "anomaly_percentage": round(100.0 * anomalies.sum() / len(mse_seq), 3),
    "threshold_95pct": float(threshold)
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "TCN_LABEVENTS_Summary.csv", index=False)
print(f"✅ Summary saved → {SUMMARY_DIR / 'TCN_LABEVENTS_Summary.csv'}")

# ===================== PLOT =====================
plt.figure(figsize=(12,6))
plt.title("TCN Autoencoder Reconstruction Error - LABEVENTS")
plt.plot(mse_seq, label="Reconstruction Error", alpha=0.7)
plt.axhline(threshold, color="orange", linestyle="--", label="Anomaly Threshold")
plt.scatter(np.where(anomalies)[0], mse_seq[anomalies], color="red", marker="x", label="Anomalies")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_TCN_AnomalyPlot.png")
plt.close()
print(f"✅ Plot saved → {OUTPUT_DIR / 'LABEVENTS_TCN_AnomalyPlot.png'}")

print("\n🎯 TCN-based Anomaly Detection completed successfully!")
