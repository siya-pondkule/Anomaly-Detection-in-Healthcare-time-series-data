import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\One-class-SVM-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)


print(f"\n=== Processing {DATA_PATH.name} for One-Class SVM Anomaly Detection ===")

# ======================
# Load and preprocess data
# ======================
df = pd.read_csv(DATA_PATH)

drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for One-Class SVM.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# ======================
# Scale features
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================
# Train One-Class SVM
# ======================
print("\nTraining One-Class SVM...")
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
ocsvm.fit(X_scaled)

# ======================
# Predictions
# ======================
y_pred = ocsvm.predict(X_scaled)                 # -1 = anomaly, 1 = normal
decision_scores = -ocsvm.decision_function(X_scaled)  # higher score = more anomalous

anomalies = np.where(y_pred == -1)[0]
print(f"\nDetected {len(anomalies)} anomalies out of {len(X_scaled)} records.")

# ====================================================
# MULTI-THRESHOLD METRICS (90, 95, 98 percentile)
# ====================================================
percentiles = [90, 95, 98]
summary_rows = []

# pseudo ground truth from 95th percentile
gt = (decision_scores > np.percentile(decision_scores, 95)).astype(int)

try:
    auc_value = roc_auc_score(gt, decision_scores)
except:
    auc_value = np.nan

best_f1 = -1
best_threshold_row = None

for p in percentiles:
    thr = np.percentile(decision_scores, p)
    preds = (decision_scores > thr).astype(int)

    precision = precision_score(gt, preds, zero_division=0)
    recall = recall_score(gt, preds, zero_division=0)
    f1 = f1_score(gt, preds, zero_division=0)

    cm = confusion_matrix(gt, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    far = fp / (fp + tn) if (fp + tn) else 0

    row = {
        "Threshold_percentile": p,
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
        "Detected_Anomalies": preds.sum()
    }

    summary_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold_row = row.copy()

# Save multi-threshold summary
summary_df_all = pd.DataFrame(summary_rows)
summary_df_all.to_csv(SUMMARY_DIR / "OneClassSVM_ICUSTAYS_MultiThresholdSummary.csv", index=False)

# Save best threshold row
pd.DataFrame([best_threshold_row]).to_csv(SUMMARY_DIR / "OneClassSVM_ICUSTAYS_BestThreshold.csv", index=False)

print("\n=== MULTI-THRESHOLD SUMMARY ===")
print(summary_df_all)

print("\n=== BEST THRESHOLD BASED ON F1-SCORE ===")
print(best_threshold_row)

# ======================
# Save full results
# ======================
results_df = pd.DataFrame({
    "Index": np.arange(len(X_scaled)),
    "Decision_Score": decision_scores,
    "Anomaly_Label": np.where(y_pred == -1, "Anomaly", "Normal")
})
results_file = OUTPUT_DIR / "ICUSTAYS_OneClassSVM_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# ======================
# Summary (Original)
# ======================
summary_data = {
    "Total Records": len(X_scaled),
    "Detected Anomalies": len(anomalies),
    "Anomaly Percentage (%)": round(100 * len(anomalies) / len(X_scaled), 2),
    "Kernel": "RBF",
    "Nu": 0.05
}
pd.DataFrame([summary_data]).to_csv(SUMMARY_DIR / "OneClassSVM_ICUSTAYS_Summary.csv", index=False)

# ======================
# Plot — unchanged
# ======================
thr95 = np.percentile(decision_scores, 95)
plt.figure(figsize=(12, 6))
plt.plot(decision_scores, label="SVM Decision Score", color="blue", alpha=0.7)
plt.axhline(thr95, color="orange", linestyle="--", label="95% Threshold")
plt.scatter(anomalies, decision_scores[anomalies], color="red", marker="x", label="Anomalies")
plt.title("One-Class SVM Anomaly Detection - ICUSTAYS")
plt.xlabel("Index")
plt.ylabel("Decision Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_OneClassSVM_AnomalyPlot.png")
plt.close()

print(f"\nPlot saved → {OUTPUT_DIR / 'ICUSTAYS_OneClassSVM_AnomalyPlot.png'}")

print("\n🎯 One-Class SVM anomaly detection complete with multi-threshold evaluation!")
