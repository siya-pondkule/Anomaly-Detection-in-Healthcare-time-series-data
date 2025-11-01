import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ========== Paths ==========
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\Transformer-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Transformer Autoencoder ===")

# ========== Encoding & delimiter detection ==========
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
delim_candidates = [",", ";", "|", "\t"]
delim_counts = {d: sample_text.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"Detected delimiter: '{best_delim}' (counts={delim_counts})")

# ========== Load CSV ==========
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ Loaded data → shape={df.shape}")

# ========== Preprocessing ==========
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
elif "charttime" in df.columns:
    df.sort_values("charttime", inplace=True)
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60.0
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)
df["time_diff_min"] = df["time_diff_min"].fillna(0)

# select numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.insert(0, "time_diff_min")
print(f"🧮 Numeric columns used: {numeric_cols}")

df_num = df[numeric_cols].copy()
df_num = df_num.fillna(df_num.mean())  # fill missing numeric values
if df_num.empty:
    raise ValueError("❌ No usable numeric columns found for modeling.")

# ========== Scale data ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# ========== Create sequences ==========
WINDOW = 20
sequences = [X_scaled[i:i+WINDOW] for i in range(len(X_scaled)-WINDOW)]
X_seq = np.array(sequences)
print(f"✅ Created sequences: {X_seq.shape} (samples, timesteps, features)")

if X_seq.shape[0] < 10:
    raise ValueError("❌ Not enough sequences to train Transformer (too small dataset).")

# ========== Transformer Encoder Block ==========
def transformer_encoder_block(x, num_heads=4, head_dim=16, ff_dim=64, dropout_rate=0.1):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(x, x)
    attn = Dropout(dropout_rate)(attn)
    x = LayerNormalization(epsilon=1e-6)(x + attn)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1], activation="linear")(ff)
    ff = Dropout(dropout_rate)(ff)
    return LayerNormalization(epsilon=1e-6)(x + ff)

# ========== Build Transformer Autoencoder ==========
timesteps, n_features = X_seq.shape[1], X_seq.shape[2]
inputs = Input(shape=(timesteps, n_features))

x = transformer_encoder_block(inputs, 4, 16, 128, 0.1)
x = transformer_encoder_block(x, 4, 16, 128, 0.1)
encoded = GlobalAveragePooling1D()(x)
latent = Dense(64, activation="relu")(encoded)
proj = Dense(timesteps * n_features)(latent)
decoded = Reshape((timesteps, n_features))(proj)
decoded = transformer_encoder_block(decoded, 2, 16, 64, 0.05)
outputs = Dense(n_features, activation="linear")(decoded)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-3), loss="mse")
model.summary()

# ========== Train ==========
es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
history = model.fit(X_seq, X_seq, epochs=40, batch_size=64, validation_split=0.1, callbacks=[es], verbose=1)

# ========== Reconstruction error ==========
X_pred = model.predict(X_seq)
mse_seq = np.mean(np.mean(np.square(X_seq - X_pred), axis=2), axis=1)
threshold = np.percentile(mse_seq, 95)
anomalies = mse_seq > threshold
print(f"🚨 Detected {anomalies.sum()} anomalies / {len(mse_seq)} sequences (threshold={threshold:.6f})")

# ========== Save results ==========
results_df = pd.DataFrame({
    "sequence_index": np.arange(len(mse_seq)),
    "reconstruction_error": mse_seq,
    "is_anomaly": anomalies.astype(int)
})
results_df.to_csv(OUTPUT_DIR / "Transformer_Results.csv", index=False)

summary = {
    "total_sequences": len(mse_seq),
    "detected_anomalies": int(anomalies.sum()),
    "threshold_95pct": float(threshold),
    "window_size": WINDOW
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "Transformer_Summary.csv", index=False)

# ========== Plot ==========
plt.figure(figsize=(12,6))
plt.plot(mse_seq, label="Reconstruction Error")
plt.axhline(threshold, color="orange", linestyle="--", label="95th percentile threshold")
plt.scatter(np.where(anomalies)[0], mse_seq[anomalies], color="red", marker="x", label="Anomalies")
plt.xlabel("Sequence index")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Transformer_AnomalyPlot.png")
plt.close()

print("\n✅ Transformer Autoencoder training & anomaly detection completed successfully.")
