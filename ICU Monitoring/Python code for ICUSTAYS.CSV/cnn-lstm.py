import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\CNN-LSTM-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Load and preprocess data
# ======================
print(f"\n=== Processing {DATA_PATH.name} for CNN-LSTM Autoencoder ===")
df = pd.read_csv(DATA_PATH)

drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found for anomaly detection.")

print(f"Using numeric columns: {list(df_numeric.columns)}")

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================
# Create sequences
# ======================
time_steps = 5

def create_sequences(data, time_steps=5):
    seq = []
    for i in range(len(data) - time_steps):
        seq.append(data[i:(i + time_steps)])
    return np.array(seq)

X_seq = create_sequences(X_scaled, time_steps)
print(f"Data shape for CNN-LSTM: {X_seq.shape}")  # (samples, timesteps, features)

# ======================
# Build CNN-LSTM Autoencoder
# ======================
def build_cnn_lstm_autoencoder(timesteps, n_features):
    model = Sequential([
        Conv1D(64, 2, activation='relu', input_shape=(timesteps, n_features)),
        MaxPooling1D(pool_size=2),
        LSTM(64, activation='relu', return_sequences=False),
        RepeatVector(timesteps - 1),
        LSTM(64, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_cnn_lstm_autoencoder(X_seq.shape[1], X_seq.shape[2])

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_seq, X_seq[:, 1:, :],
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop]
)

# ======================
# Reconstruction Error
# ======================
X_pred = model.predict(X_seq)
mse = np.mean(np.square(X_seq[:, 1:, :] - X_pred), axis=(1, 2))

# ======================
# Ground truth = no labels available → pseudo GT (all zeros)
# ======================
gt = np.zeros(len(mse), dtype=int)

# ==========================================================
# MULTI-THRESHOLD EVALUATION (90, 95, 98)
# ==========================================================
thresholds = [90, 95, 98]
eval_rows = []
best_f1 = -1
best_row = None

for pct in thresholds:
    thr = np.percentile(mse, pct)
    y_pred = (mse >= thr).astype(int)

    # Metrics
    try:
        auc = roc_auc_score(gt, mse)
    except:
        auc = np.nan

    precision = precision_score(gt, y_pred, zero_division=0)
    recall = recall_score(gt, y_pred, zero_division=0)
    f1 = f1_score(gt, y_pred, zero_division=0)

    cm = confusion_matrix(gt, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = fp / (fp + tn) if (fp + tn) else 0

    row = {
        "Threshold_%": pct,
        "Threshold_Value": thr,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Detected_Anomalies": int(y_pred.sum())
    }
    eval_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save multi-threshold summary
multi_df = pd.DataFrame(eval_rows)
multi_df.to_csv(SUMMARY_DIR / "ICUSTAYS_CNNLSTM_MultiThresholdSummary.csv", index=False)

# Save best threshold summary
pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "ICUSTAYS_CNNLSTM_BestThreshold.csv", index=False)

print("\n=== MULTI-THRESHOLD EVALUATION (90/95/98) ===")
print(multi_df)
print("\n=== BEST THRESHOLD (By F1-Score) ===")
print(best_row)

# ==========================================================
# Apply Best Threshold
# ==========================================================
best_threshold = best_row["Threshold_Value"]
final_pred = (mse >= best_threshold).astype(int)
anomalies = np.where(final_pred == 1)[0]

# ======================
# Save main results
# ======================
results_df = pd.DataFrame({
    "Sequence_Index": np.arange(len(mse)),
    "Reconstruction_Error": mse,
    "Best_Threshold": best_threshold,
    "Anomaly_Label": final_pred
})
results_df.to_csv(OUTPUT_DIR / "ICUSTAYS_CNNLSTM_Results.csv", index=False)

# ======================
# Plot Results
# ======================
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error", alpha=0.7)
plt.scatter(anomalies, mse[anomalies], color="red", marker="x", label="Detected Anomalies")
plt.axhline(best_threshold, color="orange", linestyle="--", label=f"Best Threshold ({best_row['Threshold_%']}%)")
plt.title("CNN-LSTM Autoencoder - ICUSTAYS")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_CNNLSTM_AnomalyPlot.png")
plt.close()

print("\nPlot saved →", OUTPUT_DIR / "ICUSTAYS_CNNLSTM_AnomalyPlot.png")
print("\n✅ CNN-LSTM Autoencoder updated with 90/95/98 thresholds & best-threshold evaluation!")
