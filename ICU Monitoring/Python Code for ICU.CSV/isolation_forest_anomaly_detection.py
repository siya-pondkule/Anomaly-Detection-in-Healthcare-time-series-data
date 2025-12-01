import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\IsolationForest-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Thresholds for vital signs
# ======================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR"],
    "SBP": ["SysBP", "SBP", "SystolicBP"]
}

# ======================
# Label anomalies for physiological GT
# ======================
def label_anomalies(series, col_name, gt_array, offset):
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if series.empty:
        return []

    vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
    anomalies = []

    if vital == "HR":
        anomalies += list(series[series > THRESHOLDS["HR_tachy"]].index)
        anomalies += list(series[series < THRESHOLDS["HR_brady"]].index)

    elif vital == "SBP":
        anomalies += list(series[series > THRESHOLDS["SBP_high"]].index)
        anomalies += list(series[series < THRESHOLDS["SBP_low"]].index)

    # mark GT
    for idx in anomalies:
        if idx + offset < len(gt_array):
            gt_array[idx + offset] = 1

    return anomalies

# ======================
# Main Processing
# ======================
print(f"\n=== Isolation Forest on {DATA_PATH.name} ===")

df = pd.read_csv(DATA_PATH)
df = df[["SysBP", "Pulse"]].dropna().reset_index(drop=True)

# Physiological Ground Truth
gt_array = np.zeros(len(df))
for col in df.columns:
    label_anomalies(df[col], col, gt_array, 0)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ======================
# Train Isolation Forest
# ======================
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso.fit(X_scaled)

# model predictions
model_pred = iso.predict(X_scaled)       # -1 anomaly, 1 normal
model_pred = np.where(model_pred == -1, 1, 0)

# anomaly score (higher = more anomalous)
scores = -iso.score_samples(X_scaled)

# ======================
# Threshold evaluation: 90, 95, 98
# ======================
thresholds = [90, 95, 98]
eval_rows = []
best_f1 = -1
best_row = None

for pct in thresholds:
    thr = np.percentile(scores, pct)
    y_pred = (scores >= thr).astype(int)

    # Metrics
    try:
        auc = roc_auc_score(gt_array, scores)
    except:
        auc = np.nan

    precision = precision_score(gt_array, y_pred, zero_division=0)
    recall = recall_score(gt_array, y_pred, zero_division=0)
    f1 = f1_score(gt_array, y_pred, zero_division=0)

    cm = confusion_matrix(gt_array, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    row = {
        "Threshold_percentile": pct,
        "Threshold_value": thr,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "GT_Anomalies": int(gt_array.sum()),
        "Pred_Anomalies": int(y_pred.sum())
    }
    eval_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row

# ======================
# Save all threshold evaluations
# ======================
summary_all = pd.DataFrame(eval_rows)
summary_all.to_csv(SUMMARY_DIR / "IsolationForest_ICU_Thresholds_90_95_98.csv", index=False)

# Save best threshold
pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "IsolationForest_ICU_BestThreshold.csv", index=False)

print("\n=== BEST THRESHOLD (90/95/98) ===")
print(best_row)

# ======================
# Plot
# ======================
plt.figure(figsize=(12,6))
plt.plot(scores, label="Anomaly Score")
plt.axhline(best_row["Threshold_value"], color="orange", linestyle="--",
            label=f"Best Threshold ({best_row['Threshold_percentile']}%)")

best_pred = (scores >= best_row["Threshold_value"]).astype(int)
plt.scatter(np.where(best_pred == 1)[0], scores[best_pred == 1],
            c="red", marker="x", label="Detected Anomaly")

plt.title("Isolation Forest ICU Anomaly Detection (90/95/98 Thresholds)")
plt.xlabel("Index")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICU_IsolationForest_AnomalyPlot.png")
plt.close()

print("\nIsolation Forest anomaly detection completed successfully.")
