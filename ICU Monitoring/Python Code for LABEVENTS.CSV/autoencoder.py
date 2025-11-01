import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ====================== Paths ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\Autoencoder-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Autoencoder Anomaly Detection ===")

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
    sample_lines = "\n".join([next(f) for _ in range(10)])
delim_candidates = [",", ";", "|", "\t"]
delim_counts = {d: sample_lines.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"Detected delimiter: '{best_delim}' (counts={delim_counts})")

# ====================== Load CSV ======================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ Loaded shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ====================== Preprocessing ======================
# Parse timestamps if exist
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Create time_diff feature (in minutes)
if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"].fillna(0, inplace=True)
elif "charttime" in df.columns:
    df.sort_values("charttime", inplace=True)
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"].fillna(0, inplace=True)
else:
    df["time_diff_min"] = np.arange(len(df))

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.append("time_diff_min")

print(f"🧮 Using numeric columns: {numeric_cols}")

# Clean numeric data
df_num = df[numeric_cols].copy()
for col in df_num.columns:
    df_num[col] = pd.to_numeric(df_num[col], errors="coerce")
df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
df_num.dropna(how="all", inplace=True)

if df_num.empty:
    raise ValueError("❌ No usable numeric data found in LABEVENTS.csv.")

# ====================== Scaling ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# ====================== Autoencoder Model ======================
input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation="relu")(input_layer)
encoded = Dropout(0.2)(encoded)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(decoded)
decoded = Dropout(0.2)(decoded)
output_layer = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

# ====================== Train Model ======================
es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    shuffle=True,
    callbacks=[es],
    verbose=1
)

# ====================== Anomaly Detection ======================
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
threshold = np.percentile(mse, 95)
anomalies = mse > threshold
print(f"\n🚨 Detected {np.sum(anomalies)} anomalies out of {len(mse)} (Threshold = {threshold:.6f})")

# ====================== Save Results ======================
results_df = pd.DataFrame({
    "reconstruction_error": mse,
    "is_anomaly": anomalies.astype(int)
})
results_path = OUTPUT_DIR / "LABEVENTS_Autoencoder_Results.csv"
results_df.to_csv(results_path, index=False)
print(f"✅ Results saved to {results_path}")

summary = {
    "total_records": len(mse),
    "detected_anomalies": int(np.sum(anomalies)),
    "anomaly_percentage": round(100 * np.sum(anomalies) / len(mse), 3),
    "threshold_95pct": float(threshold)
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "Autoencoder_LABEVENTS_Summary.csv", index=False)
print(f"✅ Summary saved → {SUMMARY_DIR / 'Autoencoder_LABEVENTS_Summary.csv'}")

# ====================== Plot ======================
plt.figure(figsize=(12,6))
plt.plot(mse, label="Reconstruction Error", alpha=0.7)
plt.axhline(threshold, color='orange', linestyle='--', label="Threshold (95th %)")
plt.scatter(np.where(anomalies)[0], mse[anomalies], color='red', marker='x', label="Anomalies")
plt.title("Autoencoder Anomaly Detection - LABEVENTS")
plt.xlabel("Record Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_Autoencoder_AnomalyPlot.png")
plt.close()
print(f"✅ Plot saved → {OUTPUT_DIR / 'LABEVENTS_Autoencoder_AnomalyPlot.png'}")

print("\n🎯 Autoencoder training and anomaly detection completed successfully!")
