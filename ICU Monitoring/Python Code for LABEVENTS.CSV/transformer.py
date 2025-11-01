# transformer_labevents.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# Paths (update if needed)
# -----------------------
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\Transformer-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for Transformer Autoencoder ===")

# -----------------------
# Encoding & delimiter detection
# -----------------------
# quick encoding check
try:
    with open(DATA_PATH, "rb") as f:
        sample = f.read(4096)
    encoding = "utf-8"
    sample.decode(encoding)
except Exception:
    encoding = "latin1"
print(f"Detected encoding: {encoding}")

# detect delimiter from a few lines
with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    lines = []
    for _ in range(10):
        try:
            lines.append(next(f))
        except StopIteration:
            break
sample_text = "\n".join(lines)
delim_candidates = [",", ";", "|", "\t"]
delim_counts = {d: sample_text.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"Detected delimiter: '{best_delim}' (counts={delim_counts})")

# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"Loaded data shape: {df.shape}")

# -----------------------
# Preprocessing
# -----------------------
# Convert time-like columns to datetime if present
for c in df.columns:
    if "time" in c.lower() or "date" in c.lower():
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        except Exception:
            pass

# Create time_diff_min feature to give model temporal spacing info
if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"].fillna(0.0, inplace=True)
elif "charttime" in df.columns:
    df.sort_values("charttime", inplace=True)
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"].fillna(0.0, inplace=True)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

# Pick numeric columns automatically (prefer 'valuenum' or 'value' if present)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
preferred = [c for c in ["valuenum", "value", "time_diff_min"] if c in df.columns]
# ensure time_diff_min present and first
if "time_diff_min" in df.columns:
    if "time_diff_min" in numeric_cols:
        numeric_cols = ["time_diff_min"] + [c for c in numeric_cols if c != "time_diff_min"]
    else:
        numeric_cols.insert(0, "time_diff_min")

print(f"Using numeric columns for modeling: {numeric_cols}")

# create numeric dataframe and clean it
df_num = df[numeric_cols].copy()
# coerce any numeric-like strings
for col in df_num.columns:
    df_num[col] = pd.to_numeric(df_num[col], errors="coerce")

# replace infs and drop rows that are entirely NaN (keep rows with partial data)
df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
# If many NaNs, fill with column mean (you prefer drop entirely? we fill to keep more data)
df_num.fillna(df_num.mean(), inplace=True)
# final safety: if still NaN (column fully NA), drop that column
df_num = df_num.loc[:, df_num.apply(pd.Series.nunique) > 1]

if df_num.empty:
    raise ValueError("No usable numeric columns found for modeling. Inspect LABEVENTS.csv.")

print(f"Final feature set ({df_num.shape[1]} columns): {list(df_num.columns)}")
print(f"Total numeric rows: {len(df_num)}")

# -----------------------
# Scaling
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# -----------------------
# Create sequences (sliding windows)
# -----------------------
WINDOW = 20   # change if desired (10-50 typical)
sequences = []
for i in range(0, len(X_scaled) - WINDOW + 1):
    sequences.append(X_scaled[i : i + WINDOW])
X_seq = np.array(sequences)
print(f"Created sequences shape: {X_seq.shape} (num_sequences, timesteps, features)")

if X_seq.shape[0] < 10:
    raise ValueError("Not enough sequences created. Reduce WINDOW or inspect dataset length.")

# -----------------------
# Transformer encoder block
# -----------------------
from tensorflow.keras.layers import InputLayer

def transformer_encoder_block(x, num_heads=4, head_dim=16, ff_dim=64, dropout_rate=0.1):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(x, x)
    attn = Dropout(dropout_rate)(attn)
    x = LayerNormalization(epsilon=1e-6)(x + attn)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1], activation="linear")(ff)
    ff = Dropout(dropout_rate)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    return x

# -----------------------
# Build Transformer Autoencoder
# -----------------------
timesteps, n_features = X_seq.shape[1], X_seq.shape[2]
inputs = Input(shape=(timesteps, n_features))

# encoder (2 blocks)
x = transformer_encoder_block(inputs, num_heads=4, head_dim=16, ff_dim=128, dropout_rate=0.1)
x = transformer_encoder_block(x, num_heads=4, head_dim=16, ff_dim=128, dropout_rate=0.1)

# global pooling to get compact representation
encoded = GlobalAveragePooling1D()(x)

# bottleneck
latent = Dense(64, activation="relu")(encoded)
latent = Dropout(0.1)(latent)

# decoder - project back to sequence space
proj = Dense(timesteps * n_features, activation="linear")(latent)
proj = Dropout(0.1)(proj)
decoded = Reshape((timesteps, n_features))(proj)

# optional small transformer block on decoded output
decoded = transformer_encoder_block(decoded, num_heads=2, head_dim=16, ff_dim=64, dropout_rate=0.05)

outputs = Dense(n_features, activation="linear")(decoded)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
model.summary()

# -----------------------
# Train
# -----------------------
es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
history = model.fit(
    X_seq, X_seq,
    epochs=60,
    batch_size=128,
    validation_split=0.1,
    callbacks=[es],
    verbose=1,
    shuffle=True
)

# -----------------------
# Reconstruction & anomaly scoring
# -----------------------
X_pred = model.predict(X_seq)
mse_seq = np.mean(np.mean(np.square(X_seq - X_pred), axis=2), axis=1)  # per-sequence MSE

# thresholding
threshold = np.percentile(mse_seq, 95)
anomaly_mask = mse_seq > threshold
print(f"Detected {int(anomaly_mask.sum())} anomalous sequences out of {len(mse_seq)} (threshold={threshold:.6f})")

# map sequence indices back to original row indices (start index of each sequence)
sequence_start_indices = np.arange(0, len(df_num) - WINDOW + 1)

# -----------------------
# Save results
# -----------------------
results_df = pd.DataFrame({
    "sequence_start_index": sequence_start_indices,
    "reconstruction_error": mse_seq,
    "is_anomaly": anomaly_mask.astype(int)
})
results_file = OUTPUT_DIR / "LABEVENTS_Transformer_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

summary = {
    "total_sequences": int(len(mse_seq)),
    "detected_anomalies": int(anomaly_mask.sum()),
    "anomaly_percentage": float(100.0 * anomaly_mask.sum() / len(mse_seq)),
    "window_size": int(WINDOW),
    "threshold_95pct": float(threshold)
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "Transformer_LABEVENTS_Summary.csv", index=False)
print(f"Summary saved → {SUMMARY_DIR / 'Transformer_LABEVENTS_Summary.csv'}")

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(12,6))
plt.plot(mse_seq, label="Reconstruction error (sequence MSE)")
plt.axhline(threshold, color="orange", linestyle="--", label="95th percentile threshold")
plt.scatter(np.where(anomaly_mask)[0], mse_seq[anomaly_mask], color="red", marker="x", label="Anomalies")
plt.xlabel("Sequence index")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_Transformer_AnomalyPlot.png")
plt.close()
print(f"Plot saved → {OUTPUT_DIR / 'LABEVENTS_Transformer_AnomalyPlot.png'}")

print("\n✅ Transformer training + anomaly detection finished.")
