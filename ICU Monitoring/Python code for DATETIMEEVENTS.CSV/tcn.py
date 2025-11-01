import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import chardet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import os

# ============================================================
# Paths
# ============================================================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\TCN")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for TCN-based Anomaly Detection ===")

# ============================================================
# Detect encoding and delimiter
# ============================================================
with open(DATA_PATH, "rb") as f:
    enc = chardet.detect(f.read())["encoding"]
encoding = enc or "utf-8"
print(f"✅ Encoding: {encoding}")

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    sample = f.read(2048)
best_delim = "," if sample.count(",") > sample.count(";") else ";"
print(f"🔍 Delimiter detected: '{best_delim}'")

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ Loaded data → shape={df.shape}")

# ============================================================
# Preprocess
# ============================================================
# Convert all numeric-like columns
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    except Exception:
        pass

# Handle charttime (if available)
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df.sort_values("charttime")
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

# Keep only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not numeric_cols:
    raise ValueError("❌ No usable numeric columns found in DATETIMEEVENTS.csv for TCN model.")

df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
print(f"🧮 Using numeric columns: {numeric_cols}")

# ============================================================
# Scale and create sequences
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

def create_sequences(X, seq_len=10):
    sequences = []
    for i in range(len(X) - seq_len):
        sequences.append(X[i:i+seq_len])
    return np.array(sequences)

SEQ_LEN = 10
X_seq = create_sequences(X_scaled, seq_len=SEQ_LEN)
print(f"📊 Created sequences with shape: {X_seq.shape}")

# ============================================================
# Define TCN Autoencoder
# ============================================================
input_shape = (X_seq.shape[1], X_seq.shape[2])
inputs = Input(shape=input_shape)

# Encoder
x = Conv1D(64, 2, activation="relu", padding="causal")(inputs)
x = Dropout(0.1)(x)
x = Conv1D(32, 2, activation="relu", padding="causal")(x)
x = Flatten()(x)
encoded = Dense(32, activation="relu")(x)

# Decoder
x = Dense(X_seq.shape[1] * X_seq.shape[2], activation="relu")(encoded)
x = Reshape((X_seq.shape[1], X_seq.shape[2]))(x)
x = Conv1D(32, 2, activation="relu", padding="causal")(x)
decoded = Conv1D(X_seq.shape[2], 2, activation="linear", padding="same")(x)

tcn_autoencoder = Model(inputs, decoded)
tcn_autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
tcn_autoencoder.summary()

# ============================================================
# Train Model
# ============================================================
early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

history = tcn_autoencoder.fit(
    X_seq, X_seq,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# Compute Reconstruction Error
# ============================================================
reconstructions = tcn_autoencoder.predict(X_seq)
mse = np.mean(np.square(X_seq - reconstructions), axis=(1, 2))
threshold = np.percentile(mse, 95)
labels = np.where(mse > threshold, "Anomaly", "Normal")

# ============================================================
# Save Results
# ============================================================
results_df = pd.DataFrame({
    "Index": np.arange(len(mse)),
    "Reconstruction_Error": mse,
    "Anomaly_Label": labels
})
results_file = OUTPUT_DIR / "DATETIMEEVENTS_TCN_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"\n✅ Results saved → {results_file}")

# ============================================================
# Summary
# ============================================================
summary_data = {
    "Total Records": len(mse),
    "Detected Anomalies": np.sum(labels == "Anomaly"),
    "Anomaly Percentage (%)": round(100 * np.mean(labels == "Anomaly"), 2),
    "Input Features": X_seq.shape[2],
    "Sequence Length": SEQ_LEN,
    "Threshold (95th %ile)": round(threshold, 6)
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "TCN_DATETIMEEVENTS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"📄 Summary saved → {summary_file}")

# ============================================================
# Visualization
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error", color="blue", alpha=0.7)
plt.axhline(threshold, color="orange", linestyle="--", label="Threshold (95th %ile)")
plt.scatter(np.where(mse > threshold), mse[mse > threshold], color="red", marker="x", label="Anomalies")
plt.title("TCN-Based Anomaly Detection - DATETIMEEVENTS.csv")
plt.xlabel("Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DATETIMEEVENTS_TCN_AnomalyPlot.png")
plt.close()

print(f"📊 Plot saved → {OUTPUT_DIR / 'DATETIMEEVENTS_TCN_AnomalyPlot.png'}")
print("\n✅ TCN anomaly detection training completed successfully!")
