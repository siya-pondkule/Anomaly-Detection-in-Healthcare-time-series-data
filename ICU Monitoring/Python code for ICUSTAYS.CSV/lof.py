import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\LOF-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for Local Outlier Factor Anomaly Detection ===")

# ======================
# Load and preprocess data
# ======================
df = pd.read_csv(DATA_PATH)

drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for LOF.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# ======================
# Scale features
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================
# Train Local Outlier Factor
# ======================
print("\nTraining Local Outlier Factor model...")
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    metric='euclidean',
    novelty=False
)

y_pred = lof.fit_predict(X_scaled)   # -1 = anomaly
scores = -lof.negative_outlier_factor_

anomalies = np.where(y_pred == -1)[0]
print(f"\nDetected {len(anomalies)} anomalies out of {len(X_scaled)} records.")

# ====================== BASELINE SUMMARY (unchanged pipeline) ======================
base_summary = {
    "Total Records": len(X_scaled),
    "Detected Anomalies": len(anomalies),
    "Anomaly Percentage (%)": round(100 * len(anomalies) / len(X_scaled), 2),
    "Neighbors": 20,
    "Contamination": 0.05
}
pd.DataFrame([base_summary]).to_csv(SUMMARY_DIR / "LOF_ICUSTAYS_Summary.csv", index=False)
print(f"Baseline summary saved → {SUMMARY_DIR / 'LOF_ICUSTAYS_Summary.csv'}")

# =====================================================================
# 🔥 MULTI-THRESHOLD EVALUATION (90 / 95 / 98 PERCENTILE)
# =====================================================================

thresholds = [90, 95, 98]
results_rows = []

# Create pseudo-ground truth from 95th percentile
pseudo_gt = (scores >= np.percentile(scores, 95)).astype(int)

# AUC using anomaly scores
try:
    auc_value = roc_auc_score(pseudo_gt, scores)
except:
    auc_value = np.nan

best_f1 = -1
best_row = None

for q in thresholds:
    thr = np.percentile(scores, q)
    pred = (scores >= thr).astype(int)

    precision = precision_score(pseudo_gt, pred, zero_division=0)
    recall = recall_score(pseudo_gt, pred, zero_division=0)
    f1 = f1_score(pseudo_gt, pred, zero_division=0)

    cm = confusion_matrix(pseudo_gt, pred)
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
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Detected_Anomalies": pred.sum()
    }

    results_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save multi-threshold summary
multi_summary_df = pd.DataFrame(results_rows)
multi_file = SUMMARY_DIR / "LOF_ICUSTAYS_MultiThresholdSummary.csv"
multi_summary_df.to_csv(multi_file, index=False)

# Save best threshold
best_file = SUMMARY_DIR / "LOF_ICUSTAYS_BestThreshold.csv"
pd.DataFrame([best_row]).to_csv(best_file, index=False)

print("\n=== Multi-threshold summary ===")
print(multi_summary_df)

print("\n=== BEST Threshold ===")
print(best_row)

# ====================== Visualization (unchanged pipeline) ======================
threshold_95 = np.percentile(scores, 95)

plt.figure(figsize=(12, 6))
plt.plot(scores, label="LOF Score", color="blue", alpha=0.7)
plt.axhline(threshold_95, color="orange", linestyle="--", label="95th percentile threshold")
plt.scatter(anomalies, scores[anomalies], color="red", marker="x", label="LOF Anomalies")
plt.title("Local Outlier Factor (LOF) Anomaly Detection - ICUSTAYS")
plt.xlabel("Record Index")
plt.ylabel("LOF Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_LOF_AnomalyPlot.png")
plt.close()

print(f"Plot saved → {OUTPUT_DIR / 'ICUSTAYS_LOF_AnomalyPlot.png'}")
print("\n🎯 LOF anomaly detection + 90/95/98 threshold evaluation complete!")
