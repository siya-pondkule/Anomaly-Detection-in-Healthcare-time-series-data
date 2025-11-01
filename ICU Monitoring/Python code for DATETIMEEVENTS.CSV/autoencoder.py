import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import csv

# ============================================================
# Paths
# ============================================================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\Autoencoder-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Autoencoder-based Anomaly Detection ===")

# ============================================================
# Smart delimiter + encoding detection
# ============================================================
# Detect encoding
try:
    with open(DATA_PATH, 'rb') as f:
        sample = f.read(4096)
    encoding = 'utf-8'
    sample.decode(encoding)
except Exception:
    encoding = 'latin1'
print(f"✅ Detected encoding: {encoding}")

# Detect best delimiter
with open(DATA_PATH, 'r', encoding=encoding, errors='ignore') as f:
    sample = [next(f) for _ in range(10)]
sample_text = "\n".join(sample)
delim_candidates = [',', ';', '|', '\t']
delim_counts = {d: sample_text.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"🔍 Auto-detected best delimiter: '{best_delim}' (counts={delim_counts})")

df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine='python', on_bad_lines='skip')
print(f"✅ File loaded → Shape: {df.shape}")

# ============================================================
# Preprocessing
# ============================================================
# Convert charttime to datetime
if 'charttime' in df.columns:
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')

# Sort per patient
if 'subject_id' in df.columns and 'charttime' in df.columns:
    df.sort_values(by=['subject_id', 'charttime'], inplace=True)

# Derive time-based numerical features
if 'subject_id' in df.columns and 'charttime' in df.columns:
    df['time_diff_min'] = df.groupby('subject_id')['charttime'].diff().dt.total_seconds() / 60
    df['time_diff_min'] = df['time_diff_min'].fillna(0)

# Count events per patient
if 'subject_id' in df.columns:
    event_counts = df.groupby('subject_id').size().rename('event_count')
    mean_timediff = df.groupby('subject_id')['time_diff_min'].mean().rename('avg_time_gap_min')
    df_features = pd.concat([event_counts, mean_timediff], axis=1).reset_index()
else:
    # Fallback: if no subject_id, use global time intervals
    df['time_diff_min'] = df['charttime'].diff().dt.total_seconds() / 60
    df['time_diff_min'] = df['time_diff_min'].fillna(0)
    df_features = df[['time_diff_min']].copy()

print(f"🧮 Derived features → {list(df_features.columns)}")
df_features = df_features.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)

if df_features.empty:
    raise ValueError("❌ No usable numeric columns found in DATETIMEEVENTS.csv for Autoencoder training.")

# ============================================================
# Scaling
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# ============================================================
# Define Autoencoder model
# ============================================================
input_dim = X_scaled.shape[1]
encoding_dim = max(2, input_dim // 2)

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim * 2, activation='relu')(input_layer)
encoded = Dropout(0.1)(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
decoded = Dropout(0.1)(decoded)
output_layer = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# ============================================================
# Train Autoencoder
# ============================================================
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# Compute reconstruction error & detect anomalies
# ============================================================
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
threshold = np.percentile(mse, 95)
labels = np.where(mse > threshold, "Anomaly", "Normal")

# Save results
results_df = pd.DataFrame({
    "Index": np.arange(len(df_features)),
    "Reconstruction_Error": mse,
    "Anomaly_Label": labels
})
results_file = OUTPUT_DIR / "DATETIMEEVENTS_Autoencoder_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"💾 Results saved → {results_file}")

# Summary
summary_data = {
    "Total Records": len(df_features),
    "Detected Anomalies": int(np.sum(labels == "Anomaly")),
    "Anomaly Percentage (%)": round(100 * np.mean(labels == "Anomaly"), 2),
    "Input Features": input_dim,
    "Latent Dimension": encoding_dim,
    "Threshold (95th Percentile)": round(threshold, 6)
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "Autoencoder_DATETIMEEVENTS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"📄 Summary saved → {summary_file}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error", color="blue", alpha=0.7)
plt.axhline(threshold, color="orange", linestyle="--", label="Threshold (95th %ile)")
plt.scatter(np.where(mse > threshold), mse[mse > threshold], color="red", marker="x", label="Anomalies")
plt.title("Autoencoder Anomaly Detection - DATETIMEEVENTS.csv")
plt.xlabel("Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DATETIMEEVENTS_Autoencoder_AnomalyPlot.png")
plt.close()

print(f"📊 Plot saved → {OUTPUT_DIR / 'DATETIMEEVENTS_Autoencoder_AnomalyPlot.png'}")
print("\n✅ Autoencoder training and anomaly detection completed successfully!")
