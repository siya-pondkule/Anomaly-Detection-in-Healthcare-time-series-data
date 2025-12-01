import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# Paths (adjust if necessary)
# ============================================================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\CNN-LSTM-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for CNN-LSTM-based Anomaly Detection (with metrics) ===")

# ============================================================
# Auto-detect encoding and delimiter
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
    sample_lines = []
    for _ in range(10):
        try:
            sample_lines.append(next(f))
        except StopIteration:
            break
sample_text = "\n".join(sample_lines)
delim_candidates = [',', ';', '|', '\t']
delim_counts = {d: sample_text.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"🔍 Auto-detected best delimiter: '{best_delim}' (counts={delim_counts})")

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine='python', on_bad_lines='skip')
print(f"✅ File loaded → Shape: {df.shape}")

# ============================================================
# Feature engineering
# ============================================================
# Derive a time-difference feature 'time_diff_min' (same approach used previously)
if 'charttime' in df.columns:
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')

# If subject_id & charttime exist, compute per-row time_diff; else global diff
if 'subject_id' in df.columns and 'charttime' in df.columns:
    df.sort_values(by=['subject_id', 'charttime'], inplace=True)
    df['time_diff_min'] = df.groupby('subject_id')['charttime'].diff().dt.total_seconds() / 60
    df['time_diff_min'] = df['time_diff_min'].fillna(0)
    # build subject-level features for creating an interpretable ground-truth
    event_counts = df.groupby('subject_id').size().rename('event_count')
    mean_timediff = df.groupby('subject_id')['time_diff_min'].mean().rename('avg_time_gap_min')
    subj_features = pd.concat([event_counts, mean_timediff], axis=1).reset_index()
    # merge back so each row has its subject's aggregated stats
    df = df.merge(subj_features, on='subject_id', how='left')
else:
    # fallback: compute simple time_diff on the whole file
    if 'charttime' in df.columns:
        df['time_diff_min'] = df['charttime'].diff().dt.total_seconds() / 60
        df['time_diff_min'] = df['time_diff_min'].fillna(0)
    else:
        # if no time information, create a placeholder small constant to avoid empty feature
        df['time_diff_min'] = 0.0
    # create pseudo subject-level stats (global)
    df['event_count'] = 1
    df['avg_time_gap_min'] = df['time_diff_min']

# Keep only numerical features we will use for sequence modeling
feature_col = 'time_diff_min'
if feature_col not in df.columns:
    raise ValueError(f"Required feature '{feature_col}' not found in the dataset.")

df_feat = df[[feature_col, 'event_count', 'avg_time_gap_min']].reset_index(drop=True)

# ============================================================
# Create ground-truth (y_true) for evaluation
# Strategy:
#  - mark subjects (or rows) as "ground-truth anomaly" if their subject-level metrics are extreme:
#    event_count > mean + 2*std OR avg_time_gap_min > mean + 2*std
#  - if no subject grouping available, fall back to row-level time_diff threshold (mean+2*std)
# ============================================================
gt_per_row = np.zeros(len(df_feat), dtype=int)

if 'subject_id' in df.columns:
    # compute subject-level thresholds
    ec_mean, ec_std = df_feat['event_count'].mean(), df_feat['event_count'].std(ddof=0)
    td_mean, td_std = df_feat['avg_time_gap_min'].mean(), df_feat['avg_time_gap_min'].std(ddof=0)
    ec_thresh = ec_mean + 2 * (ec_std if not np.isnan(ec_std) else 0)
    td_thresh = td_mean + 2 * (td_std if not np.isnan(td_std) else 0)

    # find subject ids that exceed thresholds
    subject_anom_mask = (df_feat['event_count'] > ec_thresh) | (df_feat['avg_time_gap_min'] > td_thresh)
    # subject_anom_mask is per-row because we merged subject-level columns
    gt_per_row = subject_anom_mask.astype(int).values
else:
    # row-level anomaly: large time gaps are considered anomalies
    td_mean, td_std = df_feat['time_diff_min'].mean(), df_feat['time_diff_min'].std(ddof=0)
    td_thresh = td_mean + 2 * (td_std if not np.isnan(td_std) else 0)
    gt_per_row = (df_feat['time_diff_min'] > td_thresh).astype(int).values

n_pos = int(gt_per_row.sum())
print(f"Ground-truth construction: {n_pos} positive rows (anomalous) out of {len(gt_per_row)}")

# ============================================================
# Prepare data for CNN-LSTM (sequence creation)
# ============================================================
# scale only the feature used for modeling (time_diff_min)
scaler = StandardScaler()
scaled_feature = scaler.fit_transform(df_feat[[feature_col]].values)  # shape (N, 1)

def create_sequences_single_feature(arr, seq_len=10):
    sequences = []
    for i in range(len(arr) - seq_len):
        sequences.append(arr[i:i + seq_len])
    return np.array(sequences)

SEQ_LEN = 10
X = create_sequences_single_feature(scaled_feature, SEQ_LEN)  # shape (num_seq, seq_len, 1)

# Build sequence-level ground-truth: if any row in the sequence is gt==1 then sequence is positive
gt_sequences = []
for i in range(len(scaled_feature) - SEQ_LEN):
    seq_gt = gt_per_row[i:i + SEQ_LEN]
    gt_sequences.append(int(seq_gt.any()))
y_true = np.array(gt_sequences)  # shape (num_seq, )

if len(X) == 0:
    raise ValueError("Not enough rows to form one sequence. Reduce SEQ_LEN or provide more data.")

print(f"Prepared sequences → X.shape={X.shape} | y_true.shape={y_true.shape} | Positive sequences={y_true.sum()}")

# Split (train on first 80% sequences)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_test = y_true[train_size:]

# ============================================================
# Define CNN-LSTM Autoencoder (same architecture as before)
# ============================================================
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(SEQ_LEN, 1)),
    MaxPooling1D(pool_size=2, padding='same'),
    LSTM(64, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=False),
    RepeatVector(SEQ_LEN),
    LSTM(32, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# ============================================================
# Train model
# ============================================================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, X_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# Reconstruction on test set -> sequence-level MSE scores
# ============================================================
X_test_pred = model.predict(X_test)
mse_seq = np.mean(np.square(X_test_pred - X_test), axis=(1, 2))  # shape (num_test_seq,)

# ============================================================
# ----------------- MULTI-THRESHOLD EVALUATION -----------------
# Evaluate at 90%, 95%, 98%, pick best by F1, save summary
# ============================================================
thresholds = {
    "90%": np.percentile(mse_seq, 90),
    "95%": np.percentile(mse_seq, 95),
    "98%": np.percentile(mse_seq, 98)
}

all_threshold_results = []
best_threshold_result = None

for name, thr in thresholds.items():
    y_pred_thr = (mse_seq > thr).astype(int)

    # AUC uses continuous values (same for all thresholds)
    try:
        auc_thr = roc_auc_score(y_test, mse_seq) if len(np.unique(y_test)) > 1 else float("nan")
    except:
        auc_thr = float("nan")

    precision_thr = precision_score(y_test, y_pred_thr, zero_division=0)
    recall_thr = recall_score(y_test, y_pred_thr, zero_division=0)
    f1_thr = f1_score(y_test, y_pred_thr, zero_division=0)

    # False Alarm Rate
    cm_thr = confusion_matrix(y_test, y_pred_thr)
    if cm_thr.size == 4:
        tn, fp, fn, tp = cm_thr.ravel()
    else:
        tn = cm_thr[0, 0] if cm_thr.size == 1 else 0
        fp = fn = tp = 0
    far_thr = fp / (fp + tn) if (fp + tn) > 0 else 0

    result = {
        "Threshold": name,
        "Threshold_Value": float(thr),
        "AUC": float(auc_thr),
        "Precision": float(precision_thr),
        "Recall": float(recall_thr),
        "F1": float(f1_thr),
        "FAR": float(far_thr)
    }

    all_threshold_results.append(result)

    # Select best threshold by highest F1-score
    if best_threshold_result is None or f1_thr > best_threshold_result["F1"]:
        best_threshold_result = result

# Print all threshold results
print("\n================ Multi-Threshold Evaluation ================")
for r in all_threshold_results:
    print(r)

print("\n================ BEST Threshold (Based on F1) ================")
print(best_threshold_result)

# Save threshold summary CSV
threshold_summary_df = pd.DataFrame(all_threshold_results)
summary_file_multi = SUMMARY_DIR / "CNN_LSTM_DATETIMEEVENTS_MultiThreshold_Summary.csv"
threshold_summary_df.to_csv(summary_file_multi, index=False)
print(f"📄 Multi-threshold summary saved → {summary_file_multi}")

# ============================================================
# After multi-threshold evaluation, use the BEST threshold to set final y_pred
# (This keeps the rest of your original pipeline behavior consistent)
# ============================================================
best_thr_value = best_threshold_result["Threshold_Value"]
y_pred = (mse_seq > best_thr_value).astype(int)

# ============================================================
# Compute metrics (use continuous scores=mse_seq for AUC)
# ============================================================
# If y_test has no positive samples, AUC is not defined -> handle gracefully
try:
    auc = float(roc_auc_score(y_test, mse_seq)) if (len(np.unique(y_test)) > 1) else float('nan')
except Exception:
    auc = float('nan')

# For classification metrics (precision, recall, f1, FAR)
precision = float(precision_score(y_test, y_pred, zero_division=0))
recall = float(recall_score(y_test, y_pred, zero_division=0))
f1 = float(f1_score(y_test, y_pred, zero_division=0))
# confusion matrix: tn, fp, fn, tp
cm = confusion_matrix(y_test, y_pred)
if cm.size == 4:
    tn, fp, fn, tp = cm.ravel()
else:
    tn = cm[0, 0] if cm.size == 1 else 0
    fp = fn = tp = 0
far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

print("\n=== Evaluation (sequence-level) ===")
print(f"AUC: {auc}")
print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | FAR: {far:.4f}")
print(f"Threshold (BEST chosen): {best_thr_value:.6f} (from {best_threshold_result['Threshold']})")

# ============================================================
# Save predictions & per-sequence summary
# ============================================================
seq_indices = np.arange(len(X_test))
results_df = pd.DataFrame({
    "Sequence_Index": seq_indices,
    "Reconstruction_Error": mse_seq,
    "Predicted_Label": y_pred,
    "GroundTruth_Label": y_test
})
results_file = OUTPUT_DIR / "DATETIMEEVENTS_CNN_LSTM_Results_with_metrics.csv"
results_df.to_csv(results_file, index=False)
print(f"💾 Results saved → {results_file}")

# ============================================================
# Save summary CSV (requested fields)
# ============================================================
summary_data = {
    "Total Sequences": int(len(X)),
    "Detected Anomalies (Predicted)": int(y_pred.sum()),
    "Anomaly Percentage Predicted (%)": round(100 * (y_pred.sum() / len(y_pred)) if len(y_pred) > 0 else 0, 3),
    "Sequence Length": SEQ_LEN,
    "Threshold (BEST chosen)": float(best_thr_value),
    "Threshold_Name": best_threshold_result['Threshold'],
    "AUC": auc,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "FAR": far
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "CNN_LSTM_DATETIMEEVENTS_Summary_with_metrics.csv"
summary_df.to_csv(summary_file, index=False)
print(f"📄 Summary (with metrics) saved → {summary_file}")

# ============================================================
# Visualization (reconstruction error plot and marked anomalies)
# ============================================================
plt.figure(figsize=(12, 6))
# plot the sequence-wise mse over the test set indices shifted to global sequence index
global_seq_indices = np.arange(train_size, train_size + len(mse_seq))
plt.plot(global_seq_indices, mse_seq, label="Reconstruction Error (test sequences)", color="blue", alpha=0.7)
plt.axhline(best_thr_value, color="orange", linestyle="--", label=f"Best Threshold ({best_threshold_result['Threshold']})")
anomaly_seq_global = global_seq_indices[y_pred == 1]
plt.scatter(anomaly_seq_global, mse_seq[y_pred == 1], color="red", marker="x", label="Predicted Anomaly")
plt.title("CNN-LSTM Reconstruction Error (sequence-level) - DATETIMEEVENTS")
plt.xlabel("Sequence (global index)")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = OUTPUT_DIR / "DATETIMEEVENTS_CNN_LSTM_AnomalyPlot_with_metrics.png"
plt.savefig(plot_path)
plt.close()
print(f"📊 Plot saved → {plot_path}")

print("\n✅ CNN-LSTM (DATETIMEEVENTS) processing completed. Summary and detailed results include AUC, Precision, Recall, F1, FAR.")
