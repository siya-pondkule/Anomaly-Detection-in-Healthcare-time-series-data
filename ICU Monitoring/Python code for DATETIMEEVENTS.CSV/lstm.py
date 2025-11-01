import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import csv

# ============================================================
# Paths
# ============================================================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\LSTM-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for LSTM-based Anomaly Detection ===")

# ============================================================
# Smart Encoding & Delimiter Detection
# ============================================================
try:
    with open(DATA_PATH, 'rb') as f:
        sample = f.read(4096)
    encoding = 'utf-8'
    sample.decode(encoding)
except Exception:
    encoding = 'latin1'
print(f"✅ Detected encoding: {encoding}")

with open(DATA_PATH, 'r', encoding=encoding, errors='ignore') as f:
    sample = [next(f) for _ in range(10)]
sample_text = "\n".join(sample)
delim_candidates = [',', ';', '|', '\t']
delim_counts = {d: sample_text.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"🔍 Auto-detected best delimiter: '{best_delim}'")

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine='python', on_bad_lines='skip')
print(f"✅ File loaded → Shape: {df.shape}")

# ============================================================
# Preprocess for LSTM
# ============================================================
if 'charttime' in df.columns:
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')

if 'subject_id' in df.columns and 'charttime' in df.columns:
    df.sort_values(by=['subject_id', 'charttime'], inplace=True)
    df['time_diff_min'] = df.groupby('subject_id')['charttime'].diff().dt.total_seconds() / 60
    df['time_diff_min'] = df['time_diff_min'].fillna(0)
else:
    df['time_diff_min'] = df['charttime'].diff().dt.total_seconds() / 60
    df['time_diff_min'] = df['time_diff_min'].fillna(0)

df_lstm = df[['time_diff_min']].dropna().reset_index(drop=True)
print(f"🧮 Using feature: {list(df_lstm.columns)}")

# ============================================================
# Scale and create sequences
# ============================================================
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_lstm)

def create_sequences(data, seq_len=10):
    sequences = []
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len]
        sequences.append(seq)
    return np.array(sequences)

SEQ_LEN = 10
X = create_sequences(scaled_data, SEQ_LEN)
print(f"📏 LSTM input shape: {X.shape}")

# ============================================================
# Define LSTM Autoencoder model
# ============================================================
model = Sequential([
    LSTM(64, activation='relu', input_shape=(SEQ_LEN, X.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=False),
    RepeatVector(SEQ_LEN),
    LSTM(32, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X.shape[2]))
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# ============================================================
# Train LSTM Autoencoder
# ============================================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X, X,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# Compute reconstruction error
# ============================================================
X_pred = model.predict(X)
mse = np.mean(np.mean(np.square(X_pred - X), axis=2), axis=1)
threshold = np.percentile(mse, 95)
labels = np.where(mse > threshold, "Anomaly", "Normal")

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame({
    "Index": np.arange(len(mse)),
    "Reconstruction_Error": mse,
    "Anomaly_Label": labels
})
results_file = OUTPUT_DIR / "DATETIMEEVENTS_LSTM_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"💾 Results saved → {results_file}")

# ============================================================
# Summary
# ============================================================
summary_data = {
    "Total Sequences": len(mse),
    "Detected Anomalies": int(np.sum(labels == "Anomaly")),
    "Anomaly Percentage (%)": round(100 * np.mean(labels == "Anomaly"), 2),
    "Sequence Length": SEQ_LEN,
    "Threshold (95th Percentile)": round(threshold, 6)
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "LSTM_DATETIMEEVENTS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"📄 Summary saved → {summary_file}")

# ============================================================
# Visualization
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error", color="blue", alpha=0.7)
plt.axhline(threshold, color="orange", linestyle="--", label="Threshold (95th %ile)")
plt.scatter(np.where(mse > threshold), mse[mse > threshold], color="red", marker="x", label="Anomalies")
plt.title("LSTM Autoencoder Anomaly Detection - DATETIMEEVENTS.csv")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DATETIMEEVENTS_LSTM_AnomalyPlot.png")
plt.close()

print(f"📊 Plot saved → {OUTPUT_DIR / 'DATETIMEEVENTS_LSTM_AnomalyPlot.png'}")
print("\n✅ LSTM anomaly detection training completed successfully!")
