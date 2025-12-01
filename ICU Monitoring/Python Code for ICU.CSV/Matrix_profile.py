import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from stumpy import stump

# =========================
# Configuration / Paths
# =========================
INPUT_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv")
RESULTS_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\Matrix-Profile")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_MULTI = RESULTS_DIR / "MatrixProfile_MultiThreshold_Summary.csv"
SUMMARY_BEST = RESULTS_DIR / "MatrixProfile_BestThreshold.csv"
RESULTS_CSV = RESULTS_DIR / "MatrixProfile_Anomaly_Results.csv"
PLOT_PATH = RESULTS_DIR / "MatrixProfile_Anomaly_Plot.png"

# =========================
# Physiological thresholds
# =========================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR"],
    "SBP": ["SysBP", "SBP", "SystolicBP"]
}


# =========================
# Ground truth labeling
# =========================
def label_anomalies_from_thresholds(df, features):
    gt = np.zeros(len(df), dtype=int)

    def mark(series, vital_key):
        series = pd.to_numeric(series, errors="coerce").reset_index(drop=True)
        if vital_key == "HR":
            return series[(series > THRESHOLDS["HR_tachy"]) | (series < THRESHOLDS["HR_brady"])].index
        elif vital_key == "SBP":
            return series[(series > THRESHOLDS["SBP_high"]) | (series < THRESHOLDS["SBP_low"])].index
        return []

    for col in features:
        vital = next((v for v, cols in VITAL_MAPPING.items() if col in cols), None)
        if vital is None: continue
        idxs = mark(df[col], vital)
        gt[idxs] = 1

    return gt


# =========================
# 1) Load & prepare data
# =========================
print("\n=== Matrix Profile anomaly detection (SysBP & Pulse) ===")
df = pd.read_csv(INPUT_PATH)
FEATURES = ["SysBP", "Pulse"]
df = df[FEATURES].dropna().reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

signal = X_scaled[:, 0]   # Use SysBP signal

# =========================
# 2) Compute Matrix Profile
# =========================
window_size = 20
mp = stump(signal, m=window_size)
matrix_profile_vals = mp[:, 0]

# Align scores to original length
scores = np.full(len(signal), np.nan)
scores[:len(matrix_profile_vals)] = matrix_profile_vals

# =========================
# 3) Ground-truth labels
# =========================
gt = label_anomalies_from_thresholds(df, FEATURES)

# =========================
# 4) Eval for 90 / 95 / 98 percentiles
# =========================
THRESHOLDS_TO_TEST = [90, 95, 98]
results = []
best_f1 = -1
best_row = None
best_preds = None
best_thr_value = None

valid_mask = ~np.isnan(scores)
scores_valid = scores[valid_mask]
gt_valid = gt[valid_mask]

try:
    auc_overall = roc_auc_score(gt_valid, scores_valid)
except:
    auc_overall = np.nan

for pct in THRESHOLDS_TO_TEST:
    thr = np.percentile(scores_valid, pct)
    preds = (scores_valid >= thr).astype(int)

    precision = precision_score(gt_valid, preds, zero_division=0)
    recall = recall_score(gt_valid, preds, zero_division=0)
    f1 = f1_score(gt_valid, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(gt_valid, preds).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0

    row = {
        "Threshold(%)": pct,
        "Threshold_Value": thr,
        "AUC": auc_overall,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Detected_Anomalies": int(preds.sum())
    }
    results.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row
        best_preds = preds
        best_thr_value = thr

# Build full-length prediction array (aligning with NaN region)
pred_full = np.zeros_like(scores, dtype=int)
pred_full[valid_mask] = best_preds

# =========================
# 5) Save Results
# =========================
pd.DataFrame(results).to_csv(SUMMARY_MULTI, index=False)
pd.DataFrame([best_row]).to_csv(SUMMARY_BEST, index=False)

df_results = pd.DataFrame({
    "Index": np.arange(len(signal)),
    "SysBP": df["SysBP"],
    "Pulse": df["Pulse"],
    "MatrixProfileScore": scores,
    "GT_Anomaly": gt,
    "Predicted_Anomaly_Best": pred_full
})
df_results.to_csv(RESULTS_CSV, index=False)

# =========================
# 6) Plot with Best Threshold
# =========================
plt.figure(figsize=(14, 6))
plt.plot(df["SysBP"], label="SysBP", alpha=0.8)
best_positions = np.where(pred_full == 1)[0]
plt.scatter(best_positions, df["SysBP"].iloc[best_positions], color="red", s=30, label="Detected Anomaly")

gt_positions = np.where(gt == 1)[0]
plt.scatter(gt_positions, df["SysBP"].iloc[gt_positions], facecolors='none', edgecolors='green', s=40, label="GT Anomaly")

plt.title(f"Matrix Profile Anomaly Detection (Best Threshold = {best_row['Threshold(%)']}%)")
plt.xlabel("Index")
plt.ylabel("SysBP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
plt.close()

# =========================
# Final Output
# =========================
print("\n=== Multi-Threshold Evaluation Complete ===")
print(pd.DataFrame(results))
print("\n=== BEST THRESHOLD ===")
print(best_row)

print(f"\nSummary (all thresholds): {SUMMARY_MULTI}")
print(f"Best threshold summary: {SUMMARY_BEST}")
print(f"Plot saved: {PLOT_PATH}")
print(f"Detailed results: {RESULTS_CSV}")
