import pandas as pd
import numpy as np
import chardet
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

# ===================== PATHS =====================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\OneClassSVM-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for One-Class SVM Anomaly Detection ===")

# ===================== ENCODING & DELIMITER DETECTION =====================
with open(DATA_PATH, "rb") as f:
    raw = f.read(4096)
encoding = chardet.detect(raw)["encoding"]
print(f"Detected encoding: {encoding}")

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    sample = [next(f) for _ in range(10)]
sample_text = "\n".join(sample)
delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: sample_text.count(d))
print(f"Detected delimiter: '{best_delim}'")

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"Loaded dataset shape: {df.shape}")

# ===================== PREPROCESSING =====================
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
else:
    df["time_diff_min"] = np.arange(len(df))

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Using numeric columns: {numeric_cols}")

df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
if df_numeric.empty:
    raise ValueError("No numeric rows found after cleaning.")

# ===================== SCALING =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)
print(f"Data scaled successfully: {X_scaled.shape}")

# ===================== TRAIN ONE-CLASS SVM =====================
print("\nTraining One-Class SVM...")
ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
ocsvm.fit(X_scaled)

# ===================== PREDICTIONS =====================
y_pred_raw = ocsvm.predict(X_scaled)
scores = -ocsvm.decision_function(X_scaled)   # higher = more anomalous

# Convert -1 to 1 (anomaly), 1 to 0 (normal)
y_pred = (y_pred_raw == -1).astype(int)

df_numeric["anomaly_flag"] = y_pred
df_numeric["anomaly_score"] = scores

print(f"Detected {y_pred.sum()} anomalies out of {len(y_pred)} records.")

# ===================== MULTI-THRESHOLD EVALUATION =====================
thresholds = [90, 95, 98]
eval_rows = []
best_row, best_f1 = None, -1

# Pseudo-ground truth = original OneClassSVM anomalies (weak supervision)
gt = y_pred.copy()

# AUC once outside loop
try:
    auc_value = roc_auc_score(gt, scores)
except:
    auc_value = np.nan

for q in thresholds:
    thr = np.percentile(scores, q)
    pred_q = (scores >= thr).astype(int)

    precision = precision_score(gt, pred_q, zero_division=0)
    recall = recall_score(gt, pred_q, zero_division=0)
    f1 = f1_score(gt, pred_q, zero_division=0)

    cm = confusion_matrix(gt, pred_q)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = (0, 0, 0, 0)

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

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
        "Detected_Anomalies": pred_q.sum()
    }
    eval_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save multi-threshold summary
multi_df = pd.DataFrame(eval_rows)
summary_file_all = SUMMARY_DIR / "OneClassSVM_LABEVENTS_MultiThresholdSummary.csv"
multi_df.to_csv(summary_file_all, index=False)

# Save best threshold
summary_file_best = SUMMARY_DIR / "OneClassSVM_LABEVENTS_BestThreshold.csv"
pd.DataFrame([best_row]).to_csv(summary_file_best, index=False)

print("\n=== MULTI-THRESHOLD SUMMARY ===")
print(multi_df)

print("\n=== BEST THRESHOLD (Based on F1-score) ===")
print(best_row)

# ===================== SAVE RESULTS =====================
results_file = OUTPUT_DIR / "LABEVENTS_OneClassSVM_Results.csv"
df_numeric.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# ===================== VISUALIZATION =====================
plt.figure(figsize=(12,6))
plt.plot(scores, label="Anomaly Score", alpha=0.7)
plt.axhline(best_row["Threshold_value"], color="orange", linestyle="--",
            label=f"Best Threshold ({best_row['Threshold_percentile']}%)")
plt.scatter(np.where(y_pred == 1)[0], scores[y_pred == 1], color="red", marker="x", label="Original Anomalies")
plt.title("One-Class SVM Anomaly Score (LABEVENTS)")
plt.xlabel("Record Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_OneClassSVM_AnomalyPlot.png")
plt.close()

print("\n🎯 One-Class SVM with Multi-Threshold Evaluation completed successfully!")
print(f"✔ Summary saved to: {summary_file_all}")
print(f"✔ Best threshold saved to: {summary_file_best}")
