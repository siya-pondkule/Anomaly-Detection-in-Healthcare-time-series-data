import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import csv

# ====================== Paths ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\IsolationForest-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Isolation Forest Anomaly Detection ===")

# ====================== Detect Encoding & Delimiter ======================
try:
    with open(DATA_PATH, "rb") as f:
        sample = f.read(4096)
    encoding = "utf-8"
    sample.decode(encoding)
except Exception:
    encoding = "latin1"
print(f"Detected encoding: {encoding}")

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    sample_lines = "\n".join([next(f) for _ in range(10)])
delim_candidates = [",", ";", "|", "\t"]
delim_counts = {d: sample_lines.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"Detected delimiter: '{best_delim}'")

# ====================== Load CSV ======================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"Loaded data → shape: {df.shape}")

# ====================== Preprocessing ======================
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds()/60
    df["time_diff_min"].fillna(0, inplace=True)
else:
    df["time_diff_min"] = np.arange(len(df))

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.append("time_diff_min")

df_numeric = df[numeric_cols].copy()
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
df_numeric.dropna(inplace=True)

# ====================== Scale ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ====================== Train Isolation Forest ======================
print("Training Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_scaled)

# ====================== Original IF Prediction ======================
scores = -iso_forest.decision_function(X_scaled)  # higher = more anomalous
pred_raw = iso_forest.predict(X_scaled)
pred_raw = np.where(pred_raw == -1, 1, 0)

print(f"\nIsolation Forest detected {pred_raw.sum()} anomalies.")

# ====================== MULTI-THRESHOLD EVALUATION ======================
thresholds = [90, 95, 98]
results = []
best_row = None
best_f1 = -1

gt = pred_raw.copy()  # pseudo GT (because no true GT exists)

for pct in thresholds:
    thr = np.percentile(scores, pct)
    y_pred = (scores >= thr).astype(int)

    try:
        auc = roc_auc_score(gt, scores)
    except:
        auc = np.nan

    precision = precision_score(gt, y_pred, zero_division=0)
    recall = recall_score(gt, y_pred, zero_division=0)
    f1 = f1_score(gt, y_pred, zero_division=0)

    cm = confusion_matrix(gt, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    far = fp / (fp + tn) if (fp + tn) else 0

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

multi_df = pd.DataFrame(results)
multi_df.to_csv(SUMMARY_DIR / "IsolationForest_LABEVENTS_MultiThresholdSummary.csv", index=False)

pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "IsolationForest_LABEVENTS_BestThreshold.csv", index=False)

print("\n=== THRESHOLD RESULTS ===")
print(multi_df)
print("\n=== BEST THRESHOLD ===")
print(best_row)

# ====================== Build Results CSV ======================
final_pred = (scores >= best_row["Threshold_Value"]).astype(int)

results_df = pd.DataFrame({
    "anomaly_score": scores,
    "is_anomaly": final_pred
})
results_df.to_csv(OUTPUT_DIR / "LABEVENTS_IsolationForest_Results.csv", index=False)

# ====================== Plot ======================
plt.figure(figsize=(12,6))
plt.title("Isolation Forest - LABEVENTS")
plt.plot(scores, label="Anomaly Score", alpha=0.7)
plt.axhline(best_row["Threshold_Value"], color="red", linestyle="--",
            label=f"Best Threshold ({best_row['Threshold_%']}%)")
plt.scatter(np.where(final_pred)[0], scores[final_pred==1], color="red", marker="x", label="Anomalies")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_IsolationForest_AnomalyPlot.png")
plt.close()

print("\n🎯 Completed Isolation Forest with multi-threshold evaluation!")
