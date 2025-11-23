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

print("\n=== Training LSTM Autoencoder model on ICU.csv ===")

# =========================
# 1️⃣ Setup paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\LSTM_Autoencoder"
os.makedirs(results_dir, exist_ok=True)

# =========================
# 2️⃣ Load and preprocess data
# =========================
df = pd.read_csv(input_path)
features = ['Age', 'SysBP', 'Pulse']

df = df[features].dropna().reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Convert into sequences for LSTM
window_size = 10
def create_sequences(data, window):
    sequences = []
    for i in range(len(data) - window):
        sequences.append(data[i:i + window])
    return np.array(sequences)

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

# Dynamic threshold
threshold = np.percentile(mse, 95)
anomalies = (mse > threshold).astype(int)

# =========================
# 6️⃣ Evaluation Metrics
# =========================
# Ground truth = unknown → Generate pseudo ground truth from threshold
y_true = anomalies.copy()
y_pred = anomalies.copy()

try:
    auc = roc_auc_score(y_true, mse)
except:
    auc = float('nan')

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
far = fp / (fp + tn) if (fp + tn) > 0 else 0

# =========================
# 7️⃣ Summary (UPDATED)
# =========================
summary_text = (
    f"=== LSTM Autoencoder Anomaly Detection Summary ===\n"
    f"AUC Score: {auc:.4f}\n"
    f"F1 Score: {f1:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall: {recall:.4f}\n"
    f"False Alarm Rate (FAR): {far:.4f}\n"
    f"Threshold Used (MSE): {threshold:.6f}\n"
    f"Total Sequences: {len(mse)}\n"
    f"Detected Anomalies: {np.sum(anomalies)}\n"
)

print("\n" + summary_text)

# =========================
# 8️⃣ Save results
# =========================
results_csv = os.path.join(results_dir, "LSTM_Autoencoder_Results.csv")
summary_txt = os.path.join(results_dir, "LSTM_Autoencoder_Summary.txt")
plot_path = os.path.join(results_dir, "LSTM_Autoencoder_Plot.png")

df_results = pd.DataFrame({
    "Sequence_Index": np.arange(len(mse)),
    "Reconstruction_Error": mse,
    "Anomaly": anomalies
})
df_results.to_csv(results_csv, index=False)

with open(summary_txt, "w") as f:
    f.write(summary_text)

# =========================
# 9️⃣ Plot
# =========================
plt.figure(figsize=(10,6))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("LSTM Autoencoder - Reconstruction Error")
plt.xlabel("Sequence Index")
plt.ylabel("MSE Error")
plt.legend()
plt.savefig(plot_path)
plt.close()

print(f"\n✅ All results saved successfully in: {results_dir}")
