import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ====================== PATHS ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\CNN-LSTM-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for CNN-LSTM Anomaly Detection ===")

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
    lines = [next(f) for _ in range(10)]
sample_text = "\n".join(lines)
delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: sample_text.count(d))
print(f"Detected delimiter: '{best_delim}'")

# ====================== Load Data ======================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"Loaded shape: {df.shape}")

# ====================== Preprocessing ======================
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"].fillna(0.0, inplace=True)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.append("time_diff_min")

df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna(how="any")

if df_numeric.empty:
    raise ValueError("No valid numeric data for CNN-LSTM training.")

print(f"Using numeric columns: {numeric_cols}")

# ====================== Normalize ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ====================== Create Sequences ======================
TIME_STEPS = 10

def create_sequences(X, time_steps=TIME_STEPS):
    seqs = []
    for i in range(len(X) - time_steps):
        seqs.append(X[i:(i + time_steps)])
    return np.array(seqs)

X_seq = create_sequences(X_scaled)
print(f"Sequence Shape: {X_seq.shape}")

# ====================== CNN-LSTM Autoencoder ======================
model = Sequential([
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',
           input_shape=(X_seq.shape[1], X_seq.shape[2])),
    MaxPooling1D(pool_size=2, padding='same'),
    LSTM(64, activation='relu', return_sequences=False),
    RepeatVector(X_seq.shape[1]),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X_seq.shape[2]))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# ====================== Train ======================
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_seq, X_seq,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ====================== Reconstruction Error ======================
X_pred = model.predict(X_seq)
recon_error = np.mean(np.square(X_seq - X_pred), axis=(1, 2))

# Ground truth (no labels → assume all normal)
gt = np.zeros(len(recon_error), dtype=int)

# ====================== MULTI-THRESHOLD EVALUATION ======================
thresholds = [90, 95, 98]
eval_rows = []
best_f1 = -1
best_row = None

for pct in thresholds:
    thr = np.percentile(recon_error, pct)
    y_pred = (recon_error >= thr).astype(int)

    try:
        auc = roc_auc_score(gt, recon_error)
    except:
        auc = np.nan

    precision = precision_score(gt, y_pred, zero_division=0)
    recall = recall_score(gt, y_pred, zero_division=0)
    f1 = f1_score(gt, y_pred, zero_division=0)

    cm = confusion_matrix(gt, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
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

multi_df = pd.DataFrame(eval_rows)
multi_df.to_csv(SUMMARY_DIR / "LABEVENTS_CNN_LSTM_MultiThresholdSummary.csv", index=False)

pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "LABEVENTS_CNN_LSTM_BestThreshold.csv", index=False)

print("\n=== MULTI-THRESHOLD RESULTS ===")
print(multi_df)
print("\n=== BEST THRESHOLD ===")
print(best_row)

# ====================== Use Best Threshold ======================
best_threshold = best_row["Threshold_Value"]
final_pred = (recon_error >= best_threshold).astype(int)
anomaly_idx = np.where(final_pred == 1)[0]

# ====================== Save Results ======================
results_df = pd.DataFrame({
    "sequence_index": np.arange(len(recon_error)),
    "reconstruction_error": recon_error,
    "is_anomaly": final_pred
})
results_df.to_csv(OUTPUT_DIR / "LABEVENTS_CNN_LSTM_Results.csv", index=False)

# ====================== Visualization ======================
plt.figure(figsize=(12,6))
plt.title("LABEVENTS CNN-LSTM Reconstruction Error")
plt.plot(recon_error, label="Reconstruction Error", alpha=0.7)
plt.axhline(best_threshold, color="red", linestyle="--",
            label=f"Best Threshold ({best_row['Threshold_%']}%)")
plt.scatter(anomaly_idx, recon_error[anomaly_idx], color="red", marker="x", label="Anomalies")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_CNN_LSTM_AnomalyPlot.png")
plt.close()

print("\n🎯 CNN-LSTM LABEVENTS anomaly detection completed successfully!")
print(f"📌 Multi-threshold summary saved at: {SUMMARY_DIR}")
print(f"📌 Best threshold summary saved at: {SUMMARY_DIR}")
