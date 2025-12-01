import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ===================== PATHS =====================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\TCN-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for TCN-based Anomaly Detection ===")

# ===================== ENCODING & DELIMITER =====================
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

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")

# ===================== PREPROCESS =====================
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.append("time_diff_min")

df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna(how="any")
if df_numeric.empty:
    raise ValueError("No numeric data found")

# ===================== NORMALIZE =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ===================== CREATE SEQUENCES =====================
TIME_STEPS = 20

def create_sequences(X, time_steps=TIME_STEPS):
    seqs = []
    for i in range(len(X) - time_steps):
        seqs.append(X[i:(i + time_steps)])
    return np.array(seqs)

X_seq = create_sequences(X_scaled)

# ===================== BUILD TCN =====================
input_shape = (X_seq.shape[1], X_seq.shape[2])
inputs = Input(shape=input_shape)

x = Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=1)(inputs)
x = Dropout(0.2)(x)
x = Conv1D(128, 3, padding='causal', activation='relu', dilation_rate=2)(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
encoded = Dense(64, activation='relu')(x)

x = Dense(X_seq.shape[1] * X_seq.shape[2], activation='relu')(encoded)
x = Reshape((X_seq.shape[1], X_seq.shape[2]))(x)
x = Conv1D(128, 3, padding='same', activation='relu')(x)
x = Dropout(0.2)(x)
x = Conv1D(64, 3, padding='same', activation='relu')(x)
decoded = Dense(X_seq.shape[2], activation='linear')(x)

model = Model(inputs, decoded)
model.compile(optimizer=Adam(1e-3), loss='mse')

# ===================== TRAIN =====================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_seq, X_seq, epochs=40, batch_size=64, validation_split=0.1, verbose=1, callbacks=[early_stop])

# ===================== RECONSTRUCTION ERROR =====================
X_pred = model.predict(X_seq)
mse_seq = np.mean(np.mean(np.square(X_seq - X_pred), axis=2), axis=1)

# ===============================================================
# 🔥 MULTI-THRESHOLD EVALUATION (90%, 95%, 98%)
# ===============================================================
thresholds = [90, 95, 98]
eval_rows = []
best_row, best_f1 = None, -1

# Weak GT (unsupervised): Use 95% anomalies as pseudo-GT
gt = (mse_seq >= np.percentile(mse_seq, 95)).astype(int)

# Compute AUC once
try:
    auc_value = roc_auc_score(gt, mse_seq)
except:
    auc_value = np.nan

for q in thresholds:
    thr = np.percentile(mse_seq, q)
    pred = (mse_seq >= thr).astype(int)

    precision = precision_score(gt, pred, zero_division=0)
    recall = recall_score(gt, pred, zero_division=0)
    f1 = f1_score(gt, pred, zero_division=0)

    cm = confusion_matrix(gt, pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = (0, 0, 0, 0)

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

    eval_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save multi-threshold summary
multi_df = pd.DataFrame(eval_rows)
multi_file = SUMMARY_DIR / "TCN_LABEVENTS_MultiThresholdSummary.csv"
multi_df.to_csv(multi_file, index=False)

# Save best threshold
best_file = SUMMARY_DIR / "TCN_LABEVENTS_BestThreshold.csv"
pd.DataFrame([best_row]).to_csv(best_file, index=False)

print("\n=== Multi-Threshold Summary ===")
print(multi_df)

print("\n=== BEST Threshold ===")
print(best_row)

# ===================== ORIGINAL SAVING =====================
threshold_95 = np.percentile(mse_seq, 95)
anomalies = mse_seq > threshold_95

results_df = pd.DataFrame({
    "sequence_index": np.arange(len(mse_seq)),
    "reconstruction_error": mse_seq,
    "is_anomaly": anomalies.astype(int)
})
results_df.to_csv(OUTPUT_DIR / "LABEVENTS_TCN_Results.csv", index=False)

# ===================== PLOT =====================
plt.figure(figsize=(12,6))
plt.plot(mse_seq, label="Reconstruction Error")
plt.scatter(np.where(anomalies)[0], mse_seq[anomalies], color="red", marker="x", label="Anomalies")
plt.axhline(best_row["Threshold_value"], color="orange", linestyle="--",
            label=f"Best Threshold ({best_row['Threshold_percentile']}%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_TCN_AnomalyPlot.png")
plt.close()

print("\n🎯 TCN with multi-threshold evaluation completed!")
print(f"✔ Summary saved to: {multi_file}")
print(f"✔ Best threshold saved to: {best_file}")
