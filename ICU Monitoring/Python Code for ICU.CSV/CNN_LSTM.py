import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\CNN-LSTM-ModelResults")
SUMMARY_DIR = OUTPUT_DIR / "Summary"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Thresholds for vital signs
# ======================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR"],
    "SBP": ["SysBP", "SBP", "SystolicBP"]
}

# ======================
# Label anomalies
# ======================
def label_anomalies(series, col_name, gt_array, index_offset):
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if series.empty:
        return []

    vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
    anomalous_indices = []

    if vital == "HR":
        anomalous_indices += list(series[series > THRESHOLDS["HR_tachy"]].index)
        anomalous_indices += list(series[series < THRESHOLDS["HR_brady"]].index)

    elif vital == "SBP":
        anomalous_indices += list(series[series > THRESHOLDS["SBP_high"]].index)
        anomalous_indices += list(series[series < THRESHOLDS["SBP_low"]].index)

    # Mark GT array
    for i in anomalous_indices:
        if i + index_offset < len(gt_array):
            gt_array[i + index_offset] = 1

    return anomalous_indices

# ======================
# Create sequences
# ======================
def create_sequences(X, seq_len=10):
    return np.array([X[i:i+seq_len] for i in range(len(X)-seq_len)])

# ======================
# CNN-LSTM Autoencoder
# ======================
def build_cnn_lstm_autoencoder(seq_length, n_features):
    model = models.Sequential([
        layers.Input(shape=(seq_length, n_features)),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.LSTM(64, activation='relu', return_sequences=True),
        layers.LSTM(32, activation='relu', return_sequences=False),
        layers.RepeatVector(seq_length),
        layers.LSTM(32, activation='relu', return_sequences=True),
        layers.LSTM(64, activation='relu', return_sequences=True),
        layers.Conv1D(n_features, 3, activation='linear', padding='same')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ======================
# Main Processing
# ======================
print(f"\n=== Training CNN-LSTM model on {DATA_PATH.name} ===")

df = pd.read_csv(DATA_PATH)
df = df[["SysBP", "Pulse"]].dropna().reset_index(drop=True)

# Build Ground Truth (same pipeline as your original)
gt_array = np.zeros(len(df))
for col in df.columns:
    label_anomalies(df[col], col, gt_array, 0)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Sequence preparation
SEQ_LEN = 10
X_seq = create_sequences(X_scaled, SEQ_LEN)
y_seq = gt_array[SEQ_LEN:]

# Train/Test split
train_size = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_test = y_seq[train_size:]

# Train the model
model = build_cnn_lstm_autoencoder(SEQ_LEN, X_scaled.shape[1])
history = model.fit(X_train, X_train, epochs=40, batch_size=32, validation_split=0.1, verbose=1)

# Predict
X_pred = model.predict(X_test)
mse_scores = np.mean(np.square(X_pred - X_test), axis=(1, 2))

# ======================
# Threshold Evaluation (90, 95, 98)
# ======================
thresholds = [90, 95, 98]
results = []
best_f1 = -1
best_row = None

for pct in thresholds:
    thr = np.percentile(mse_scores, pct)
    y_pred = (mse_scores >= thr).astype(int)

    # Metrics
    try:
        auc = roc_auc_score(y_test, mse_scores)
    except:
        auc = np.nan

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    row = {
        "Threshold_percentile": pct,
        "Threshold_value": thr,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "GT_Anomalies": int(y_test.sum()),
        "Pred_Anomalies": int(y_pred.sum())
    }
    results.append(row)

    # Best threshold selection
    if f1 > best_f1:
        best_f1 = f1
        best_row = row

# ======================
# Save CSVs
# ======================
pd.DataFrame(results).to_csv(SUMMARY_DIR / "CNNLSTM_ICU_Thresholds_90_95_98.csv", index=False)
pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "CNNLSTM_ICU_BestThreshold.csv", index=False)

print("\n=== BEST THRESHOLD (from 90, 95, 98) ===")
print(best_row)

# ======================
# Plot MSE with threshold
# ======================
plt.figure(figsize=(12,6))
plt.plot(mse_scores, label="MSE Error")

best_thr = best_row["Threshold_value"]
plt.axhline(best_thr, color='orange', linestyle='--', label=f"Best Threshold ({best_row['Threshold_percentile']}%)")

best_pred = (mse_scores >= best_thr).astype(int)
plt.scatter(np.where(best_pred==1)[0], mse_scores[best_pred==1], c="red", marker="x", label="Detected Anomaly")

plt.title("CNN-LSTM ICU Anomaly Detection (90/95/98 Threshold Evaluation)")
plt.xlabel("Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "CNNLSTM_ICU_AnomalyPlot.png")
plt.close()

print("\nCNN-LSTM anomaly detection completed successfully.")
