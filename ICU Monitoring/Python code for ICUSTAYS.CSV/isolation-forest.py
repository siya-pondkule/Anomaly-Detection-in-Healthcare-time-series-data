import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
)

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\Isolation-Forest-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Load and preprocess data
# ======================
print(f"\n=== Processing {DATA_PATH.name} for Isolation Forest ===")
df = pd.read_csv(DATA_PATH)

drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["time", "date", "id", "unit"])]
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
numeric_df = numeric_df.dropna().reset_index(drop=True)

if numeric_df.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for Isolation Forest anomaly detection.")

print(f"Using columns for anomaly detection: {list(numeric_df.columns)}")

# ======================
# Scaling
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# ======================
# Train Isolation Forest
# ======================
print("\nTraining Isolation Forest model...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    max_samples='auto',
    n_jobs=-1
)
iso_forest.fit(X_scaled)

# Predict
predictions = iso_forest.predict(X_scaled)
scores = -iso_forest.decision_function(X_scaled)  # higher = more anomalous

print(f"Detected {np.sum(predictions==-1)} anomalies (IsolationForest internal)")

# ======================
# Save base results
# ======================
df_results = numeric_df.copy()
df_results["Anomaly_Score"] = scores
df_results["Anomaly_Label"] = np.where(predictions == -1, 1, 0)

results_file = OUTPUT_DIR / "ICUSTAYS_IsolationForest_Results.csv"
df_results.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# =======================================================
# 🔥 MULTI-THRESHOLD EVALUATION (90 / 95 / 98)
# =======================================================

thresholds = [90, 95, 98]
eval_rows = []
best_f1 = -1
best_row = None

# Use 95th percentile threshold as pseudo ground-truth (unsupervised)
pseudo_gt = (scores >= np.percentile(scores, 95)).astype(int)

# AUC using anomaly scores
try:
    auc_value = roc_auc_score(pseudo_gt, scores)
except:
    auc_value = np.nan

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

    eval_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save Multi-threshold summary
multi_df = pd.DataFrame(eval_rows)
multi_df_file = SUMMARY_DIR / "IsolationForest_ICUSTAYS_MultiThresholdSummary.csv"
multi_df.to_csv(multi_df_file, index=False)

# Save Best Threshold
best_df_file = SUMMARY_DIR / "IsolationForest_ICUSTAYS_BestThreshold.csv"
pd.DataFrame([best_row]).to_csv(best_df_file, index=False)

print("\n=== Multi-threshold evaluation ===")
print(multi_df)

print("\n=== BEST THRESHOLD ===")
print(best_row)

# ======================
# Summary (original pipeline)
# ======================
summary_data = {
    "Total Records": len(df_results),
    "Detected Anomalies": int(np.sum(predictions == -1)),
    "Anomaly Percentage (%)": round(100 * np.sum(predictions == -1) / len(df_results), 2)
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "IsolationForest_ICUSTAYS_Summary.csv"
summary_df.to_csv(summary_file, index=False)

print(f"\nSummary saved → {summary_file}")
print(f"Multi-threshold summary saved → {multi_df_file}")
print(f"Best threshold summary saved → {best_df_file}")

# ======================
# Plot (unchanged)
# ======================
plt.figure(figsize=(12, 6))
plt.title("Isolation Forest Anomaly Detection - ICUSTAYS.csv")
plt.plot(scores, label="Anomaly Score", color="blue", alpha=0.7)
plt.axhline(best_row["Threshold_value"], color="orange", linestyle="--",
            label=f"Best Threshold ({best_row['Threshold_percentile']}%)")
plt.scatter(np.where(scores >= best_row["Threshold_value"])[0],
            scores[scores >= best_row["Threshold_value"]],
            color="red", marker="x", label="Anomalies")
plt.xlabel("Record Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_IsolationForest_Plot.png")
plt.close()

print("\n🎯 Isolation Forest anomaly detection + multi-threshold evaluation complete!")
