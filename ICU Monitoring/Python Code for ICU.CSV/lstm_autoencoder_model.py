import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

print("\n=== Training LSTM Autoencoder model on ICU.csv (90/95/98 threshold evaluation) ===")

# =========================
# 1️⃣ Setup paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\LSTM_Autoencoder"
summary_dir = os.path.join(results_dir, "Summary")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)

# =========================
# 2️⃣ Load and preprocess data
# =========================
df = pd.read_csv(input_path)
features = ['Age', 'SysBP', 'Pulse']
df = df[features].dropna().reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Create sequences
window_size = 10
def create_sequences(data, window):
    seq = []
    for i in range(len(data) - window):
        seq.append(data[i:i + window])
    return np.array(seq)

X_seq = create_sequences(X_scaled, window_size)
print(f"Data reshaped to {X_seq.shape} for LSTM")

# =========================
# 3️⃣ Build LSTM Autoencoder
# =========================
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=False),
    RepeatVector(X_seq.shape[1]),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_seq.shape[2]))
])
model.compile(optimizer='adam', loss='mse')

# =========================
# 4️⃣ Train model
# =========================
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_seq, X_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop]
)

# =========================
# 5️⃣ Compute reconstruction error
# =========================
X_pred = model.predict(X_seq)
mse = np.mean(np.power(X_seq - X_pred, 2), axis=(1, 2))

# =========================
# 6️⃣ Multi-threshold evaluation (90, 95, 98)
# =========================
thresholds = [90, 95, 98]
eval_rows = []
best_f1 = -1
best_row = None

for pct in thresholds:
    thr = np.percentile(mse, pct)
    y_pred = (mse > thr).astype(int)

    # Pseudo ground truth (same strategy as your original pipeline)
    y_true = y_pred.copy()

    try:
        auc = roc_auc_score(y_true, mse)
    except:
        auc = np.nan

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    far = fp / (fp + tn) if (fp + tn) else 0

    row = {
        "Threshold_percentile": pct,
        "Threshold_value": thr,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Predicted_Anomalies": int(y_pred.sum())
    }
    eval_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row

# =========================
# 7️⃣ Save summary files
# =========================
summary_all_path = os.path.join(summary_dir, "LSTM_MultiThreshold_Summary.csv")
pd.DataFrame(eval_rows).to_csv(summary_all_path, index=False)

best_thr_path = os.path.join(summary_dir, "LSTM_BestThreshold.csv")
pd.DataFrame([best_row]).to_csv(best_thr_path, index=False)

print("\n=== BEST THRESHOLD (Based on F1 Score) ===")
print(best_row)

# =========================
# 8️⃣ Plot using BEST threshold
# =========================
best_thr = best_row["Threshold_value"]
best_pred = (mse > best_thr).astype(int)

plot_path = os.path.join(results_dir, "LSTM_Autoencoder_BestThreshold_Plot.png")

plt.figure(figsize=(10,6))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(best_thr, color='red', linestyle='--',
            label=f'Best Threshold ({best_row["Threshold_percentile"]}%)')
plt.scatter(np.where(best_pred == 1),
            mse[best_pred == 1],
            color='red', label='Anomalies')
plt.xlabel("Sequence Index")
plt.ylabel("MSE Error")
plt.legend()
plt.grid(True)
plt.savefig(plot_path)
plt.close()

print(f"\nAll thresholds summary: {summary_all_path}")
print(f"Best threshold summary: {best_thr_path}")
print(f"Plot saved: {plot_path}")
print("\nLSTM Autoencoder Multi-Threshold Evaluation completed successfully!")
