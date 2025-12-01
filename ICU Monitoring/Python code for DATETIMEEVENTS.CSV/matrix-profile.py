import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stumpy
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# =========================
# Config / Paths
# =========================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\MatrixProfile-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Matrix Profile (DATETIMEEVENTS) → {DATA_PATH} ===")

# =========================
# Smart read: detect encoding + delimiter
# =========================
def detect_encoding(path, nbytes=4096):
    try:
        with open(path, "rb") as f:
            sample = f.read(nbytes)
        sample.decode("utf-8")
        return "utf-8"
    except Exception:
        return "latin1"

def detect_delimiter(path, encoding, nlines=10):
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        lines = []
        for _ in range(nlines):
            try:
                lines.append(next(f))
            except StopIteration:
                break
    text = "\n".join(lines)
    candidates = [",", ";", "|", "\t"]
    counts = {c: text.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best, counts

encoding = detect_encoding(DATA_PATH)
delim, delim_counts = detect_delimiter(DATA_PATH, encoding)
print(f"Detected encoding: {encoding} | delimiter: '{delim}' counts={delim_counts}")

# Load CSV robustly
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=delim, engine="python", on_bad_lines="skip")
print(f"Loaded dataframe shape: {df.shape}")

# =========================
# Preprocess / feature selection
# =========================
# Convert charttime if present
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

# Create time_diff_min if possible
if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
elif "charttime" in df.columns:
    df.sort_values("charttime", inplace=True)
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
else:
    # fallback index-based synthetic time_diff
    df["time_diff_min"] = np.arange(len(df)).astype(float)

# Preferential column list to use for matrix profile & for physiological GT
preferred_cols = ["value", "valuenum", "SysBP", "Pulse", "time_diff_min"]
selected_col = None
for c in preferred_cols:
    if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
        selected_col = c
        break

# If none of the preferred columns exist, choose the first numerical column
if selected_col is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in DATETIMEEVENTS.csv to analyze.")
    selected_col = numeric_cols[0]

series = pd.to_numeric(df[selected_col], errors="coerce").fillna(method="ffill").fillna(0.0).astype(float).values
series_length = len(series)
print(f"Using column '{selected_col}' for Matrix Profile (length={series_length})")

# =========================
# Create ground truth labels (y_true)
# Strategy:
# - If SysBP/Pulse available, use physiological thresholds (Option A).
# - Else, use time_diff_min threshold (mean + 3*std) as anomalies.
# - Else, fallback: mark top 5% largest values in selected_col as "physiological" anomalies.
# The true labels are at the original index resolution (one label per sample).
# Matrix profile produces values for subsequence starts; we will align later.
# =========================
gt = np.zeros(series_length, dtype=int)

if "Pulse" in df.columns or "SysBP" in df.columns:
    # SysBP
    if "SysBP" in df.columns:
        sbp = pd.to_numeric(df["SysBP"], errors="coerce")
        gt[(sbp > 140).fillna(False).values] = 1
        gt[(sbp < 90).fillna(False).values] = 1
    # Pulse
    if "Pulse" in df.columns:
        pulse = pd.to_numeric(df["Pulse"], errors="coerce")
        gt[(pulse > 100).fillna(False).values] = 1
        gt[(pulse < 60).fillna(False).values] = 1
    method_used = "physiological_thresholds"
elif "time_diff_min" in df.columns:
    t = pd.to_numeric(df["time_diff_min"], errors="coerce").fillna(0.0)
    thr = t.mean() + 3 * t.std()
    gt[(t > thr).values] = 1
    method_used = "time_gap_>mean+3std"
else:
    # fallback: label top 5% highest values as anomalies
    pct = 95
    thr = np.percentile(series, pct)
    gt[(series >= thr)] = 1
    method_used = f"top_{100-pct}pct_values"

print(f"Ground-truth labeling method: {method_used} | Total GT anomalies: {int(gt.sum())}")

# =========================
# Compute Matrix Profile
# =========================
# Choose window m (must be < series_length). Use sensible default.
m = min(50, max(3, series_length // 10))
if series_length < m * 2:
    # reduce window if series is short
    m = max(3, series_length // 4)
print(f"Using subsequence window (m) = {m}")

mp = stumpy.stump(series, m=m)   # returns array of shape (n - m + 1, 2)
matrix_profile = mp[:, 0]        # profile values (higher -> more anomalous for discords in our usage)
mp_idx = np.arange(len(matrix_profile))

# Align ground truth to subsequence starts:
# For each subsequence start i, consider it anomalous if any of the m samples in that subsequence are GT=1.
gt_subseq = np.array([1 if gt[i:i+m].sum() > 0 else 0 for i in mp_idx])

# Scores: matrix_profile (higher -> more anomalous)
scores = matrix_profile.copy()

# =========================
# Compute continuous AUC (same for all thresholds since it uses continuous scores)
# =========================
try:
    auc_score_continuous = roc_auc_score(gt_subseq, scores)
except Exception:
    auc_score_continuous = np.nan

# =========================
# NEW: Evaluate multiple thresholds (90, 95, 98) and pick best among them (C option)
# Replace previous single 95% behaviour with evaluation only on these percentiles.
# =========================
threshold_percentiles = [90, 95, 98]
thresholds_summary = []

for pct in threshold_percentiles:
    thr = float(np.percentile(scores, pct))
    y_pred_thr = (scores >= thr).astype(int)

    # compute metrics
    try:
        prec = precision_score(gt_subseq, y_pred_thr, zero_division=0)
    except Exception:
        prec = 0.0
    try:
        rec = recall_score(gt_subseq, y_pred_thr, zero_division=0)
    except Exception:
        rec = 0.0
    try:
        f1 = f1_score(gt_subseq, y_pred_thr, zero_division=0)
    except Exception:
        f1 = 0.0

    cm = confusion_matrix(gt_subseq, y_pred_thr)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    thresholds_summary.append({
        "percentile": pct,
        "threshold": thr,
        "AUC_continuous": float(auc_score_continuous) if not np.isnan(auc_score_continuous) else np.nan,
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "FAR": float(far),
        "Predicted_Anomalies": int(y_pred_thr.sum())
    })

# Convert to DataFrame
thresholds_df = pd.DataFrame(thresholds_summary)

# =========================
# Choose best threshold among these three using:
#  - primary: highest F1
#  - tie-breaker 1: higher Precision
#  - tie-breaker 2: higher Recall
# =========================
best_row = thresholds_df.sort_values(
    by=["F1", "Precision", "Recall"], ascending=False
).iloc[0]

best_percentile = int(best_row["percentile"])
best_threshold = float(best_row["threshold"])
best_metrics = best_row.to_dict()

print("\n=== Per-threshold evaluation ===")
print(thresholds_df.to_string(index=False, float_format="%.6f"))

print("\n=== Best threshold chosen from [90,95,98] (by F1, then Precision, then Recall) ===")
print(f"Best percentile: {best_percentile} | Threshold: {best_threshold:.6f}")
print(f"Metrics (AUC_continuous, Precision, Recall, F1, FAR, Predicted): "
      f"{best_metrics['AUC_continuous']:.6f}, {best_metrics['Precision']:.6f}, "
      f"{best_metrics['Recall']:.6f}, {best_metrics['F1']:.6f}, {best_metrics['FAR']:.6f}, "
      f"{int(best_metrics['Predicted_Anomalies'])}")

# =========================
# Final predicted labels using the best threshold
# =========================
y_pred_best = (scores >= best_threshold).astype(int)

# Recompute confusion matrix and metrics for the best threshold (to ensure consistency)
cm = confusion_matrix(gt_subseq, y_pred_best)
if cm.size == 4:
    tn, fp, fn, tp = cm.ravel()
else:
    tn = fp = fn = tp = 0
precision_best = precision_score(gt_subseq, y_pred_best, zero_division=0)
recall_best = recall_score(gt_subseq, y_pred_best, zero_division=0)
f1_best = f1_score(gt_subseq, y_pred_best, zero_division=0)
far_best = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

# =========================
# Save detailed results (per subsequence start) — using best threshold label
# =========================
results_df = pd.DataFrame({
    "subseq_start_index": mp_idx,
    "matrix_profile_value": matrix_profile,
    "gt_subseq": gt_subseq,
    "pred_label_best": y_pred_best,
    "score": scores
})
results_file = OUTPUT_DIR / "DATETIMEEVENTS_MatrixProfile_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# =========================
# Save summary CSV with per-threshold rows + best selection
# Fields:
# - Per-threshold rows: percentile, threshold, AUC_continuous, Precision, Recall, F1, FAR, Predicted_Anomalies
# - A final summary row indicating best chosen threshold & metrics
# =========================
# Per-threshold summary file
thresholds_out_file = SUMMARY_DIR / "MatrixProfile_DATETIMEEVENTS_Thresholds_Evaluation.csv"
thresholds_df.to_csv(thresholds_out_file, index=False)
print(f"Per-threshold evaluation saved → {thresholds_out_file}")

# Overall summary
summary = {
    "Total Records": int(series_length),
    "Window Size": int(m),
    "Total GT Anomalies (points)": int(gt.sum()),
    "Total GT Anomalies (subseq aligned)": int(gt_subseq.sum()),
    "Detected Anomalies (predicted_subseq_best)": int(y_pred_best.sum()),
    "Anomaly Percentage (pred %)": round(100 * (int(y_pred_best.sum()) / len(y_pred_best)), 6) if len(y_pred_best) > 0 else 0.0,
    "Best_Percentile": int(best_percentile),
    "Best_Threshold": float(round(best_threshold, 6)),
    "AUC_continuous": float(round(auc_score_continuous, 6)) if not np.isnan(auc_score_continuous) else np.nan,
    "Precision_best": float(round(precision_best, 6)),
    "Recall_best": float(round(recall_best, 6)),
    "F1_best": float(round(f1_best, 6)),
    "FAR_best": float(round(far_best, 6)),
    "Top_anomaly_subseq_start_index": int(np.argmax(scores)),
    "Top_anomaly_score": float(round(float(np.max(scores)), 6)),
    "GT_Labeling_Method": method_used
}
summary_df = pd.DataFrame([summary])
summary_file = SUMMARY_DIR / "MatrixProfile_DATETIMEEVENTS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved → {summary_file}")

# =========================
# Plots
# 1) Time series with detected subsequence anomalies (mark center of subsequence)
# 2) Matrix profile with best threshold & GT marked
# 3) ROC curve (if possible)
# =========================
# Plot 1: Time series + detected anomaly centers
plt.figure(figsize=(14, 5))
plt.plot(series, label=f"{selected_col}", linewidth=1)
# compute subsequence center indices for plotting
centers = mp_idx + m // 2
pred_centers = centers[y_pred_best == 1]
gt_centers = centers[gt_subseq == 1]
plt.scatter(pred_centers, series[pred_centers], c='red', s=30, marker='x', label='Predicted Anomaly (subseq center - best thr)')
plt.scatter(gt_centers, series[gt_centers], c='green', s=20, marker='o', facecolors='none', label='GT Anomaly (subseq center)')
plt.title("DATETIMEEVENTS — Time Series with Detected Matrix Profile Anomalies (Best Threshold)")
plt.xlabel("Index")
plt.ylabel(selected_col)
plt.legend()
plt.grid(True)
plt.tight_layout()
plot1 = OUTPUT_DIR / "DATETIMEEVENTS_MatrixProfile_TimeSeries_Anomalies_BestThr.png"
plt.savefig(plot1, dpi=200)
plt.close()
print(f"Plot saved → {plot1}")

# Plot 2: Matrix profile values with best threshold
plt.figure(figsize=(14, 4))
plt.plot(mp_idx, scores, label="Matrix Profile (score)")
plt.axhline(best_threshold, color="orange", linestyle="--", label=f"Best Threshold (p={best_percentile})")
plt.scatter(mp_idx[gt_subseq==1], scores[gt_subseq==1], c='green', s=20, label='GT subseq (anomaly)')
plt.scatter(mp_idx[y_pred_best==1], scores[y_pred_best==1], c='red', s=20, marker='x', label='Predicted subseq (anomaly - best thr)')
plt.title("Matrix Profile — DATETIMEEVENTS (Best Threshold Shown)")
plt.xlabel("Subsequence start index")
plt.ylabel("Profile value (score)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot2 = OUTPUT_DIR / "DATETIMEEVENTS_MatrixProfile_ProfilePlot_BestThr.png"
plt.savefig(plot2, dpi=200)
plt.close()
print(f"Plot saved → {plot2}")

# Plot 3: ROC Curve (if possible)
if not np.all(gt_subseq == 0) and not np.all(gt_subseq == 1):
    fpr, tpr, _ = roc_curve(gt_subseq, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Matrix Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot3 = OUTPUT_DIR / "DATETIMEEVENTS_MatrixProfile_ROC.png"
    plt.savefig(plot3, dpi=200)
    plt.close()
    print(f"Plot saved → {plot3}")
else:
    print("ROC plot skipped (need both positive and negative GT labels).")

print("\n✅ Matrix Profile (DATETIMEEVENTS) processing complete.")
