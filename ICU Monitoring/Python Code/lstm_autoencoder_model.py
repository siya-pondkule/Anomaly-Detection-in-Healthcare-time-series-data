import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

print("\n=== Training LSTM Autoencoder model on ICU.csv ===")

# =========================
# 1️⃣ Setup paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models/LSTM_Autoencoder"
os.makedirs(results_dir, exist_ok=True)

# =========================
# 2️⃣ Load and preprocess data
# =========================
df = pd.read_csv(input_path)
features = ['Age', 'SysBP', 'Pulse']  # same columns used in LOF + IsolationForest

df = df[features].dropna().reset_index(drop=True)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Convert to 3D for LSTM (samples, timesteps, features)
# Using sliding window with window size = 10
window_size = 10
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        seq = data[i : i + window_size]
        sequences.append(seq)
    return np.array(sequences)

X_seq = create_sequences(X_scaled, window_size)
print(f"Data reshaped to {X_seq.shape} for LSTM training")

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
# 5️⃣ Compute reconstruction errors
# =========================
X_pred = model.predict(X_seq)
mse = np.mean(np.power(X_seq - X_pred, 2), axis=(1,2))

# Set threshold for anomalies
threshold = np.mean(mse) + 3 * np.std(mse)
anomalies = mse > threshold

# =========================
# 6️⃣ Summary
# =========================
num_anomalies = np.sum(anomalies)
summary_text = (
    f"=== LSTM Autoencoder Anomaly Detection Summary ===\n"
    f"Total sequences: {len(mse)}\n"
    f"Detected anomalies: {num_anomalies}\n"
    f"Normal sequences: {len(mse) - num_anomalies}\n"
    f"Threshold (MSE): {threshold:.6f}\n"
)

print("\n" + summary_text)

# =========================
# 7️⃣ Save results
# =========================
df_results = pd.DataFrame({
    "Sequence_Index": np.arange(len(mse)),
    "Reconstruction_Error": mse,
    "Anomaly": anomalies.astype(int)
})

results_csv = os.path.join(results_dir, "LSTM_Autoencoder_Results.csv")
summary_txt = os.path.join(results_dir, "LSTM_Autoencoder_Summary.txt")
plot_path = os.path.join(results_dir, "LSTM_Autoencoder_Plot.png")

df_results.to_csv(results_csv, index=False)
with open(summary_txt, "w") as f:
    f.write(summary_text)

# =========================
# 8️⃣ Plot
# =========================
plt.figure(figsize=(10,6))
plt.plot(mse, label='Reconstruction Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.title("LSTM Autoencoder Anomaly Detection")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 9️⃣ Final message
# =========================
print(f"✅ LSTM Autoencoder results saved successfully in: {results_dir}")
print(f"   - CSV: {results_csv}")
print(f"   - Summary: {summary_txt}")
print(f"   - Plot: {plot_path}")
