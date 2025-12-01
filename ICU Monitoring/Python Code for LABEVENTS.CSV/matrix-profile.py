import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stumpy
from pathlib import Path
import chardet
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ===================== PATHS =====================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\MATRIX-PROFILE-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for Matrix Profile Anomaly Detection ===")

# ===================== ENCODING DETECTION =====================
with open(DATA_PATH, "rb") as f:
    rawdata = f.read(4096)
encoding = chardet.detect(rawdata)["encoding"]
print(f"Detected encoding: {encoding}")

# ===================== DELIMITER DETECTION =====================
with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    sample_lines = [next(f) for _ in range(10)]
sample_text = "\n".join(sample_lines)
delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: sample_text.count(d))
print(f"Detected delimiter: '{best_delim}'")

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine='python', on_bad_lines="skip")
print(f"Loaded LABEVENTS.csv → Shape: {df.shape}")

# ===================== PREPROCESSING =====================
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds()/60
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
else:
    df["time_diff_min"] = np.arange(len(df))

value_col = None
for col in ["valuenum", "value"]:
    if col in df.columns:
        value_col = col
        break

if not value_col:
    raise ValueError("No numeric value column found!")

df = df[[value_col, "time_diff_min"]].dropna()

series = df[value_col].astype(float).values

print(f"Using feature: {value_col} | Total records: {len(series)}")

# ===================== MATRIX PROFILE =====================
window_size = max(10, len(series)//100)
print(f"Computing Matrix Profile (window size = {window_size})...")

mp = stumpy.stump(series, m=window_size)
matrix_profile = mp[:, 0]

# Pad to full length
if len(matrix_profile) < len(series):
    matrix_profile = np.pad(matrix_profile, (0, len(series)-len(matrix_profile)),
                            mode="constant", constant_values=np.nan)

# ===== Top discords (kept same as your pipeline) =====
n_anomalies = min(10, len(series))
discord_idx = np.argsort(-matrix_profile[:len(series)])[:n_anomalies]

is_anomaly = np.zeros(len(series), dtype=int)
is_anomaly[discord_idx] = 1

# Save Base Results
results_df = pd.DataFrame({
    "index": np.arange(len(series)),
    value_col: series,
    "matrix_profile": matrix_profile,
    "top_discords_flag": is_anomaly
})
results_df.to_csv(OUTPUT_DIR / "LABEVENTS_MatrixProfile_Results.csv", index=False)

print(f"Top anomalies: {discord_idx.tolist()}")

# ============================================================
#   MULTI-THRESHOLD EVALUATION (90, 95, 98) — ADDED BY REQUEST
# ============================================================

thresholds = [90, 95, 98]
eval_rows = []
best_row, best_f1 = None, -1

# Use top-discords as pseudo ground truth (weak supervision)
pseudo_gt = is_anomaly.copy()

for q in thresholds:
    thr = np.nanpercentile(matrix_profile, q)
    y_pred = (matrix_profile >= thr).astype(int)

    # AUC score
    try:
        auc = roc_auc_score(pseudo_gt, matrix_profile)
    except:
        auc = np.nan

    precision = precision_score(pseudo_gt, y_pred, zero_division=0)
    recall = recall_score(pseudo_gt, y_pred, zero_division=0)
    f1 = f1_score(pseudo_gt, y_pred, zero_division=0)

    cm = confusion_matrix(pseudo_gt, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    far = fp / (fp + tn) if (fp + tn) else 0

    row = {
        "Threshold_%": q,
        "Threshold_Value": thr,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Detected_Anomalies": y_pred.sum()
    }
    eval_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save evaluation summary
multi_df = pd.DataFrame(eval_rows)
multi_df.to_csv(SUMMARY_DIR / "MatrixProfile_LABEVENTS_MultiThresholdSummary.csv", index=False)

pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "MatrixProfile_LABEVENTS_BestThreshold.csv", index=False)

print("\n=== MULTI-THRESHOLD SUMMARY ===")
print(multi_df)

print("\n=== BEST THRESHOLD ===")
print(best_row)

# ===================== VISUALIZE =====================
plt.figure(figsize=(12,6))
plt.plot(series, label=value_col)
plt.scatter(discord_idx, series[discord_idx], color="red", marker="x", label="Original Discords")
plt.axhline(best_row["Threshold_Value"], color="orange", linestyle="--",
            label=f"Best Threshold ({best_row['Threshold_%']}%)")
plt.legend()
plt.grid(True)
plt.savefig(OUTPUT_DIR / "LABEVENTS_MatrixProfile_AnomalyPlot.png")
plt.close()

print("\n🎯 Matrix Profile (with multi-threshold analysis) completed successfully!")
