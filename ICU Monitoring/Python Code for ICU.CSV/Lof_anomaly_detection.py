import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

print("\n=== Training LOF anomaly detection model on ICU.csv (with 90/95/98 thresholds) ===")

# =========================
# 1️⃣ Setup paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\LOF-ModelResult"
summary_dir = os.path.join(results_dir, "Summary")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(summary_dir, exist_ok=True)

# =========================
# 2️⃣ Load dataset
# =========================
df = pd.read_csv(input_path)
df = df[['SysBP', 'Pulse']].dropna().reset_index(drop=True)

# =========================
# 3️⃣ Physiological Ground Truth
# =========================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

gt = np.zeros(len(df))

# HR anomalies
gt[df['Pulse'] > THRESHOLDS["HR_tachy"]] = 1
gt[df['Pulse'] < THRESHOLDS["HR_brady"]] = 1

# SBP anomalies
gt[df['SysBP'] > THRESHOLDS["SBP_high"]] = 1
gt[df['SysBP'] < THRESHOLDS["SBP_low"]] = 1

# =========================
# 4️⃣ Preprocessing
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================
# 5️⃣ Train LOF model
# =========================
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=False
)

raw_pred = lof.fit_predict(X_scaled)
model_pred = np.where(raw_pred == -1, 1, 0)  # model anomaly labels
scores = -lof.negative_outlier_factor_

# =========================
# 6️⃣ Threshold-based evaluation (90, 95, 98)
# =========================
thresholds = [90, 95, 98]
eval_rows = []
best_f1 = -1
best_row = None

for pct in thresholds:
    thr = np.percentile(scores, pct)
    y_pred = (scores >= thr).astype(int)

    # Metrics
    try:
        auc_val = roc_auc_score(gt, scores)
    except:
        auc_val = np.nan

    precision = precision_score(gt, y_pred, zero_division=0)
    recall = recall_score(gt, y_pred, zero_division=0)
    f1 = f1_score(gt, y_pred, zero_division=0)

    cm = confusion_matrix(gt, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    row = {
        "Threshold_percentile": pct,
        "Threshold_value": thr,
        "AUC": auc_val,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "GT_Anomalies": int(gt.sum()),
        "Pred_Anomalies": int(y_pred.sum())
    }
    eval_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row

# =========================
# 7️⃣ Save summary CSVs
# =========================
summary_all = pd.DataFrame(eval_rows)
summary_all_path = os.path.join(summary_dir, "LOF_ICU_Thresholds_90_95_98.csv")
summary_all.to_csv(summary_all_path, index=False)

best_path = os.path.join(summary_dir, "LOF_ICU_BestThreshold.csv")
pd.DataFrame([best_row]).to_csv(best_path, index=False)

print("\n=== BEST THRESHOLD (90/95/98) ===")
print(best_row)

# =========================
# 8️⃣ Plot best threshold results
# =========================
best_thr = best_row["Threshold_value"]
best_pred = (scores >= best_thr).astype(int)

plot_path = os.path.join(results_dir, "LOF_Anomaly_Plot_BestThreshold.png")

plt.figure(figsize=(8,6))
plt.scatter(df['SysBP'], df['Pulse'], c=best_pred, cmap='coolwarm')
plt.xlabel("SysBP")
plt.ylabel("Pulse")
plt.title(f"LOF Anomaly Detection (Best threshold = {best_row['Threshold_percentile']}%)")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 9️⃣ Final output
# =========================
print(f"\nResults saved in: {results_dir}")
print(f"All thresholds summary → {summary_all_path}")
print(f"Best threshold summary → {best_path}")
print(f"Plot saved → {plot_path}")
