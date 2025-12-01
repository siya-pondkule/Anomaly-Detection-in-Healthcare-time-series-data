import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Reshape, Add, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\TCN-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for TCN Autoencoder ===")
df = pd.read_csv(DATA_PATH)

# Select numeric columns only, drop IDs/timestamps
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for TCN anomaly detection.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================
# Create sequences
# ======================
time_steps = 10
def create_sequences(data, time_steps=10):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

X_seq = create_sequences(X_scaled, time_steps)
print(f"Shape of data for TCN: {X_seq.shape}")

# ======================
# TCN block
# ======================
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    prev_x = x
    x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)

    if prev_x.shape[-1] != filters:
        prev_x = Conv1D(filters, 1, padding="same")(prev_x)

    x = Add()([x, prev_x])
    x = Activation("relu")(x)
    return x

# ======================
# Build model
# ======================
def build_tcn_autoencoder(timesteps, n_features):
    inputs = Input(shape=(timesteps, n_features))
    x = tcn_block(inputs, 64, 3, dilation_rate=1, dropout_rate=0.1)
    x = tcn_block(x, 64, 3, dilation_rate=2, dropout_rate=0.1)
    encoded = Flatten()(x)

    bottleneck = Dense(32, activation='relu')(encoded)

    x = Dense(timesteps * 64, activation='relu')(bottleneck)
    x = Reshape((timesteps, 64))(x)

    x = tcn_block(x, 64, 3, dilation_rate=1, dropout_rate=0.1)
    outputs = Conv1D(n_features, 1, activation='linear', padding="same")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_tcn_autoencoder(X_seq.shape[1], X_seq.shape[2])

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\nTraining TCN Autoencoder...")
history = model.fit(
    X_seq, X_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop]
)

# ======================
# Reconstruction error
# ======================
X_pred = model.predict(X_seq)
mse = np.mean(np.mean(np.square(X_seq - X_pred), axis=2), axis=1)

threshold_95 = np.percentile(mse, 95)
anomalies_95 = np.where(mse > threshold_95)[0]

print(f"\nDetected {len(anomalies_95)} anomalies at 95% threshold.")

# ========================================================
# MULTI-THRESHOLD EVALUATION (90, 95, 98)
# ========================================================

percentiles = [90, 95, 98]
summary_rows = []

# pseudo-GT using 95th percentile
gt = (mse > np.percentile(mse, 95)).astype(int)

try:
    auc_value = roc_auc_score(gt, mse)
except:
    auc_value = np.nan

best_f1 = -1
best_row = None

for p in percentiles:
    thr = np.percentile(mse, p)
    preds = (mse > thr).astype(int)

    precision = precision_score(gt, preds, zero_division=0)
    recall = recall_score(gt, preds, zero_division=0)
    f1 = f1_score(gt, preds, zero_division=0)

    cm = confusion_matrix(gt, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    far = fp / (fp + tn) if (fp + tn) else 0

    row = {
        "Percentile": p,
        "Threshold": thr,
        "AUC": auc_value,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Detected_Anomalies": preds.sum()
    }

    summary_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save full summary and best threshold
pd.DataFrame(summary_rows).to_csv(SUMMARY_DIR / "TCN_ICUSTAYS_MultiThresholdSummary.csv", index=False)
pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "TCN_ICUSTAYS_BestThreshold.csv", index=False)

print("\n=== MULTI-THRESHOLD SUMMARY ===")
print(pd.DataFrame(summary_rows))

print("\n=== BEST THRESHOLD ===")
print(best_row)

# ======================
# Save results (original)
# ======================
results_df = pd.DataFrame({
    "Sequence_Index": np.arange(len(mse)),
    "Reconstruction_Error": mse,
    "Anomaly_Label": np.where(mse > threshold_95, "Anomaly", "Normal")
})
results_file = OUTPUT_DIR / "ICUSTAYS_TCN_Results.csv"
results_df.to_csv(results_file, index=False)

# ======================
# Summary (original)
# ======================
summary_data = {
    "Total Sequences": len(mse),
    "Detected Anomalies": len(anomalies_95),
    "Anomaly Percentage (%)": round(100 * len(anomalies_95) / len(mse), 2)
}
pd.DataFrame([summary_data]).to_csv(SUMMARY_DIR / "TCN_ICUSTAYS_Summary.csv", index=False)

# ======================
# Plot
# ======================
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error")
plt.scatter(anomalies_95, mse[anomalies_95], color="red", marker="x", label="Detected Anomalies")
plt.axhline(threshold_95, color="orange", linestyle="--", label="95% Threshold")
plt.title("TCN Autoencoder Anomaly Detection - ICUSTAYS")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_TCN_Anomaly_Plot.png")
plt.close()

print(f"\nPlot saved → {OUTPUT_DIR / 'ICUSTAYS_TCN_Anomaly_Plot.png'}")
print("\n🎯 TCN Autoencoder anomaly detection with multi-threshold evaluation complete!")
