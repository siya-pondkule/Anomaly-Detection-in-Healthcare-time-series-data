import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import chardet

# ===================== PATHS =====================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\LOF-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for Local Outlier Factor (LOF) Anomaly Detection ===")

# ===================== ENCODING DETECTION =====================
with open(DATA_PATH, "rb") as f:
    rawdata = f.read(4096)
encoding = chardet.detect(rawdata)["encoding"]
print(f"Detected encoding: {encoding}")

# ===================== DELIMITER DETECTION =====================
with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    lines = [next(f) for _ in range(10)]
sample = "\n".join(lines)
delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: sample.count(d))
print(f"Detected delimiter: '{best_delim}'")

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"Loaded shape: {df.shape}")

# ===================== PREPROCESSING =====================
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds()/60
    df["time_diff_min"].fillna(0, inplace=True)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.append("time_diff_min")

df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()

print(f"Using numeric columns: {numeric_cols}")

# ===================== SCALE =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ===================== TRAIN LOF =====================
print("\nTraining LOF...")
lof = LocalOutlierFactor(
    n_neighbors=25,
    contamination=0.05,
    novelty=False
)
y_pred_raw = lof.fit_predict(X_scaled)
lof_scores = -lof.negative_outlier_factor_  # higher → more anomalous

# Convert raw LOF: -1 = anomaly → 1 (pseudo GT)
y_gt = np.where(y_pred_raw == -1, 1, 0)

print(f"LOF detected {y_gt.sum()} anomalies (raw contamination).")

# ======================================================
# 📌 MULTI-THRESHOLD EVALUATION (90, 95, 98)
# ======================================================
thresholds = [90, 95, 98]
results = []
best_row, best_f1 = None, -1

for pct in thresholds:
    thr = np.percentile(lof_scores, pct)
    y_pred = (lof_scores >= thr).astype(int)

    try:
        auc = roc_auc_score(y_gt, lof_scores)
    except:
        auc = np.nan

    precision = precision_score(y_gt, y_pred, zero_division=0)
    recall = recall_score(y_gt, y_pred, zero_division=0)
    f1 = f1_score(y_gt, y_pred, zero_division=0)

    cm = confusion_matrix(y_gt, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    far = fp / (fp + tn) if (fp + tn) else 0.0

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
    results.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save multi-threshold results
multi_df = pd.DataFrame(results)
multi_df.to_csv(SUMMARY_DIR / "LOF_LABEVENTS_MultiThresholdSummary.csv", index=False)

pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "LOF_LABEVENTS_BestThreshold.csv", index=False)

print("\n=== MULTI-THRESHOLD RESULTS ===")
print(multi_df)
print("\n=== BEST THRESHOLD BASED ON F1 ===")
print(best_row)

# ======================================================
# NORMAL LOF OUTPUT USING BEST THRESHOLD
# ======================================================
final_pred = (lof_scores >= best_row["Threshold_Value"]).astype(int)

results_df = pd.DataFrame({
    "index": df_numeric.index,
    "lof_score": lof_scores,
    "is_anomaly": final_pred
})
results_df.to_csv(OUTPUT_DIR / "LABEVENTS_LOF_Results.csv", index=False)

# ===================== PLOT =====================
plt.figure(figsize=(12,6))
plt.title("LOF Anomaly Detection - LABEVENTS")
plt.plot(lof_scores, label="LOF Score", alpha=0.7)
plt.axhline(best_row["Threshold_Value"], color="red", linestyle="--",
            label=f"Best Threshold ({best_row['Threshold_%']}%)")
plt.scatter(np.where(final_pred)[0], lof_scores[final_pred==1], color="red", marker="x", label="Anomalies")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_LOF_AnomalyPlot.png")
plt.close()

print("\n🎯 LOF anomaly detection with multi-threshold evaluation completed successfully!")
