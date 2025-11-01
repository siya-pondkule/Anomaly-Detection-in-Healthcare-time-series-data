import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import layers, models

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\CNN-LSTM-ModelResults")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Thresholds for physiological labeling
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

    for i in anomalous_indices:
        if i + index_offset < len(gt_array):
            gt_array[i + index_offset] = 1

    return [{"vital": col_name, "index": i + index_offset} for i in anomalous_indices]

# ======================
# Create sequences
# ======================
def create_sequences(X, seq_length=10):
    sequences = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
    return np.array(sequences)

# ======================
# ✅ Fixed CNN-LSTM Autoencoder (keeps same seq length)
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
# Evaluation
# ======================
def evaluate_anomaly_detection(y_true, mse_scores):
    try:
        auc = roc_auc_score(y_true, mse_scores)
    except ValueError:
        auc = np.nan

    best_f1, best_threshold = 0, np.percentile(mse_scores, 90)
    for q in np.linspace(90, 99.9, 100):
        threshold = np.percentile(mse_scores, q)
        y_pred = (mse_scores >= threshold).astype(int)
        if np.sum(y_pred) > 0 and np.sum(y_true) > 0:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold

    y_pred_best = (mse_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "AUC": auc,
        "F1": best_f1,
        "Precision": precision_score(y_true, y_pred_best, zero_division=0),
        "Recall": recall_score(y_true, y_pred_best, zero_division=0),
        "FAR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "Threshold": best_threshold
    }

# ======================
# Main
# ======================
print(f"\n=== Training CNN-LSTM model on {DATA_PATH.name} ===")
df = pd.read_csv(DATA_PATH)
df = df[["SysBP", "Pulse"]].dropna().reset_index(drop=True)

gt_array = np.zeros(len(df))
anomalies = []
for col in df.columns:
    anomalies.extend(label_anomalies(df[col], col, gt_array, 0))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

SEQ_LEN = 10
X_seq = create_sequences(X_scaled, SEQ_LEN)
y_seq = gt_array[SEQ_LEN:]

train_size = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_test = y_seq[train_size:]

model = build_cnn_lstm_autoencoder(SEQ_LEN, X_scaled.shape[1])
history = model.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
print(f"Training complete | Final Loss: {history.history['loss'][-1]:.5f}")

# Reconstruction
X_test_pred = model.predict(X_test)
mse_scores = np.mean(np.square(X_test_pred - X_test), axis=(1, 2))

# Evaluate
eval_results = evaluate_anomaly_detection(y_test, mse_scores)

print(f"\nResults for CNN-LSTM:")
print(f"AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | "
      f"Precision={eval_results['Precision']:.4f} | Recall={eval_results['Recall']:.4f} | "
      f"FAR={eval_results['FAR']:.4f}")

# Save anomalies
anomaly_indices = np.where(mse_scores >= eval_results["Threshold"])[0] + SEQ_LEN + train_size
summary = pd.DataFrame({
    "Anomaly Index": anomaly_indices,
    "Reconstruction Error": mse_scores[anomaly_indices - train_size - SEQ_LEN]
})
summary.to_csv(OUTPUT_DIR / "CNNLSTM_Anomaly_Summary.csv", index=False)
print(f"\n🧾 Summary saved → {OUTPUT_DIR / 'CNNLSTM_Anomaly_Summary.csv'}")

# Plot anomalies
plt.figure(figsize=(12,6))
plt.plot(df["SysBP"], label="SysBP", alpha=0.7)
plt.plot(df["Pulse"], label="Pulse", alpha=0.7)
for idx in anomaly_indices:
    if idx < len(df):
        plt.scatter(idx, df.loc[idx, "SysBP"], color="red", marker="x", s=50)
        plt.scatter(idx, df.loc[idx, "Pulse"], color="red", marker="x", s=50)

plt.title("ICU Anomaly Detection using CNN-LSTM (Fixed Sequence Length)")
plt.xlabel("Record Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICU_CNNLSTM_AnomalyPlot.png")
plt.close()

print(f"\n✅ Model training, evaluation, and plots completed successfully.")
