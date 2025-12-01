import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import stumpy  # library for matrix profile calculation

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\Matrix-Profile-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)


print(f"\n=== Processing {DATA_PATH.name} for Matrix Profile Anomaly Detection ===")

# ======================
# Load and preprocess data
# ======================
df = pd.read_csv(DATA_PATH)

# Remove ID/time/date/unit columns
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No numeric columns found in ICUSTAYS.csv for Matrix Profile.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# ======================
# Scale data
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# Convert to single univariate signal (mean of features)
signal = np.mean(X_scaled, axis=1)

# ======================
# Compute Matrix Profile
# ======================
window_size = 30
print(f"\nComputing Matrix Profile — window size = {window_size} ...")

mp = stumpy.stump(signal, m=window_size)
matrix_profile = mp[:, 0]

# ======================
# MULTI-THRESHOLD evaluation (90/95/98)
# ======================
thresholds = [90, 95, 98]
summary_rows = []

# Ground truth assumption: Top 5% are anomalies
gt = (matrix_profile > np.percentile(matrix_profile, 95)).astype(int)

try:
    auc_value = roc_auc_score(gt, matrix_profile)
except:
    auc_value = np.nan

best_f1 = -1
best_row = None

for q in thresholds:
    thr = np.percentile(matrix_profile, q)
    preds = (matrix_profile > thr).astype(int)

    precision = precision_score(gt, preds, zero_division=0)
    recall = recall_score(gt, preds, zero_division=0)
    f1 = f1_score(gt, preds, zero_division=0)

    cm = confusion_matrix(gt, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    row = {
        "Threshold_percentile": q,
        "Threshold_value": thr,
        "AUC": auc_value,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Detected_Anomalies": preds.sum()
    }

    summary_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save multi-threshold summary
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_DIR / "MatrixProfile_ICUSTAYS_MultiThresholdSummary.csv", index=False)

# Save best threshold
pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "MatrixProfile_ICUSTAYS_BestThreshold.csv", index=False)

print("\n=== MULTI-THRESHOLD SUMMARY ===")
print(summary_df)

print("\n=== BEST THRESHOLD (Based on F1-score) ===")
print(best_row)

# ----------------------
# Default 95% threshold for plotting (unchanged)
# ----------------------
threshold_95 = np.percentile(matrix_profile, 95)
anomalies = np.where(matrix_profile > threshold_95)[0]

# ======================
# Save results
# ======================
results_df = pd.DataFrame({
    "Index": np.arange(len(matrix_profile)),
    "Matrix_Profile_Value": matrix_profile,
    "Anomaly_Label": np.where(matrix_profile > threshold_95, "Anomaly", "Normal")
})
results_file = OUTPUT_DIR / "ICUSTAYS_MatrixProfile_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# ======================
# Summary
# ======================
summary_data = {
    "Total Points": len(matrix_profile),
    "Detected_Anomalies_95pct": len(anomalies),
    "Anomaly_Perc_95pct (%)": round(100 * len(anomalies) / len(matrix_profile), 2),
    "Window_Size": window_size
}
pd.DataFrame([summary_data]).to_csv(SUMMARY_DIR / "MatrixProfile_ICUSTAYS_Summary.csv", index=False)

# ======================
# Visualization
# ======================
plt.figure(figsize=(12, 6))
plt.plot(signal, label="ICU Signal (Mean of features)", color="blue", alpha=0.7)
plt.scatter(anomalies, signal[anomalies], color="red", marker="x", label="Detected Anomalies")
plt.title("Matrix Profile Anomaly Detection - ICUSTAYS.csv")
plt.xlabel("Index")
plt.ylabel("Scaled Signal Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_MatrixProfile_AnomalyPlot.png")
plt.close()

# Matrix Profile Plot
plt.figure(figsize=(12, 5))
plt.plot(matrix_profile, color="green", label="Matrix Profile", alpha=0.8)
plt.axhline(threshold_95, color="orange", linestyle="--", label="95% Anomaly Threshold")
plt.scatter(anomalies, matrix_profile[anomalies], color="red", marker="x", label="Anomalies")
plt.title("Matrix Profile (ICUSTAYS.csv)")
plt.xlabel("Index")
plt.ylabel("Matrix Profile Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_MatrixProfile_Plot.png")
plt.close()

print("\n🎯 Full Matrix Profile multi-threshold anomaly detection completed!")
