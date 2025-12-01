import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\LSTM-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Load Data
# ======================
print(f"\n=== Processing {DATA_PATH.name} for LSTM Autoencoder ===")
df = pd.read_csv(DATA_PATH)

# Keep only numeric columns except ID/time columns
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id","time","date","unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# ======================
# Scaling
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================
# Create Sequences
# ======================
time_steps = 5

def create_sequences(data, time_steps=5):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

X_seq = create_sequences(X_scaled, time_steps)
print(f"Shape for LSTM: {X_seq.shape}")


# ======================
# Build LSTM Autoencoder
# ======================
def build_lstm_autoencoder(timesteps, n_features):
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(timesteps, n_features), return_sequences=False),
        RepeatVector(timesteps),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_lstm_autoencoder(X_seq.shape[1], X_seq.shape[2])

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\nTraining LSTM Autoencoder...")
history = model.fit(
    X_seq, X_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop]
)

# ======================
# Compute Reconstruction Error
# ======================
X_pred = model.predict(X_seq)
mse = np.mean(np.mean(np.square(X_seq - X_pred), axis=2), axis=1)

# ======================
# Create Pseudo Ground Truth (95th percentile)
# ======================
gt = (mse >= np.percentile(mse, 95)).astype(int)

# ======================
# MULTI-THRESHOLD EVALUATION
# ======================
thresholds = [90, 95, 98]
summary_rows = []

try:
    auc_value = roc_auc_score(gt, mse)
except:
    auc_value = np.nan

best_f1 = -1
best_row = None

for q in thresholds:
    thr = np.percentile(mse, q)
    pred = (mse >= thr).astype(int)

    precision = precision_score(gt, pred, zero_division=0)
    recall = recall_score(gt, pred, zero_division=0)
    f1 = f1_score(gt, pred, zero_division=0)

    cm = confusion_matrix(gt, pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    row = {
        "Threshold_percentile": q,
        "Threshold_value": thr,
        "AUC": auc_value,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Detected_Anomalies": pred.sum()
    }

    summary_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()


summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_DIR / "LSTM_ICUSTAYS_MultiThresholdSummary.csv", index=False)
pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "LSTM_ICUSTAYS_BestThreshold.csv", index=False)

print("\n=== MULTI-THRESHOLD SUMMARY ===")
print(summary_df)

print("\n=== BEST THRESHOLD (Based on F1 Score) ===")
print(best_row)

# ======================
# Save main results
# ======================
results_df = pd.DataFrame({
    "Sequence_Index": np.arange(len(mse)),
    "Reconstruction_Error": mse
})
results_df.to_csv(OUTPUT_DIR / "ICUSTAYS_LSTM_Results.csv", index=False)

# ======================
# Plot
# ======================
thr95 = np.percentile(mse, 95)

plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error", color="blue")
plt.axhline(thr95, label="95th Percentile Threshold", color="orange", linestyle="--")
plt.scatter(np.where(mse >= thr95)[0], mse[mse >= thr95], color="red", marker="x", label="Anomalies")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.title("LSTM Autoencoder Anomaly Detection - ICUSTAYS")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_LSTM_AnomalyPlot.png")
plt.close()

print("\n🎯 LSTM Autoencoder anomaly detection + full threshold evaluation complete!")
