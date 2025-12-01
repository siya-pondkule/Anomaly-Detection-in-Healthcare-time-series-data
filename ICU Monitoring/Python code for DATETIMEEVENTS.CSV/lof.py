import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\LOF-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Local Outlier Factor (LOF) Anomaly Detection ===")

# ============================================================
# Detect encoding + delimiter
# ============================================================
try:
    with open(DATA_PATH, "rb") as f:
        f.read(4096).decode("utf-8")
    encoding = "utf-8"
except:
    encoding = "latin1"

print(f"✅ Detected encoding: {encoding}")

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    rows = [next(f) for _ in range(10)]
sample_text = "\n".join(rows)

delims = [",", ";", "|", "\t"]
delim_counts = {d: sample_text.count(d) for d in delims}
best_delim = max(delim_counts, key=delim_counts.get)

print(f"🔍 Auto-detected best delimiter: '{best_delim}'")

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ File loaded → Shape: {df.shape}")

# ============================================================
# Add time difference
# ============================================================
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60
else:
    df.sort_values("charttime", inplace=True)
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60

df["time_diff_min"] = df["time_diff_min"].fillna(0)

# ============================================================
# Create numeric feature matrix
# ============================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

df_numeric = df[numeric_cols].copy()
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
df_numeric.fillna(df_numeric.mean(), inplace=True)
df_numeric = df_numeric.loc[:, df_numeric.apply(pd.Series.nunique) > 1]

print(f"🧮 Using features: {df_numeric.columns.tolist()}")

# ============================================================
# Create physiological ground truth (GT)
# ============================================================
time_diff = df_numeric["time_diff_min"].values
median_gap = np.median(time_diff)

y_true = np.where(time_diff > 3 * median_gap, 1, 0)  # anomaly if large jump

print(f"📌 Physiological GT anomalies detected: {sum(y_true)}")

# ============================================================
# Scaling
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ============================================================
# Train LOF model
# ============================================================
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred_raw = lof.fit_predict(X_scaled)

y_pred = np.where(y_pred_raw == -1, 1, 0)
lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous

# ============================================================
# Compute AUC once (same for all thresholds)
# ============================================================
try:
    auc_all = roc_auc_score(y_true, lof_scores)
except:
    auc_all = np.nan

# ============================================================
# Evaluate at thresholds: 90%, 95%, 98%
# ============================================================
thresholds = {
    "90%": 90,
    "95%": 95,
    "98%": 98
}

threshold_results = []
best_choice = None

for name, pct in thresholds.items():
    thr_val = np.percentile(lof_scores, pct)
    y_pred_thr = (lof_scores >= thr_val).astype(int)

    precision = precision_score(y_true, y_pred_thr, zero_division=0)
    recall = recall_score(y_true, y_pred_thr, zero_division=0)
    f1 = f1_score(y_true, y_pred_thr, zero_division=0)

    cm = confusion_matrix(y_true, y_pred_thr)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        far = 0

    res = {
        "Threshold": name,
        "Percentile": pct,
        "Threshold_Value": thr_val,
        "AUC": auc_all,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "Predicted_Anomalies": int(y_pred_thr.sum())
    }
    threshold_results.append(res)

    # Best threshold selection rule: Highest F1 → tie-breaker: Precision
    if best_choice is None:
        best_choice = res
    else:
        if res["F1"] > best_choice["F1"]:
            best_choice = res
        elif res["F1"] == best_choice["F1"] and res["Precision"] > best_choice["Precision"]:
            best_choice = res

print("\n=== Threshold Evaluation (90%, 95%, 98%) ===")
for r in threshold_results:
    print(r)

print("\n🏆 Best Threshold =", best_choice["Threshold"], "| F1 =", best_choice["F1"])

# Save threshold summary
threshold_df = pd.DataFrame(threshold_results)
threshold_file = SUMMARY_DIR / "LOF_DATETIMEEVENTS_Threshold_Summary.csv"
threshold_df.to_csv(threshold_file, index=False)

# ============================================================
# FINAL METRICS using best threshold
# ============================================================
final_thr = best_choice["Threshold_Value"]
y_pred_final = (lof_scores >= final_thr).astype(int)

precision_final = precision_score(y_true, y_pred_final, zero_division=0)
recall_final = recall_score(y_true, y_pred_final, zero_division=0)
f1_final = f1_score(y_true, y_pred_final, zero_division=0)
cm = confusion_matrix(y_true, y_pred_final)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
far_final = fp / (fp + tn) if (fp + tn) > 0 else 0

# ============================================================
# Save final summary
# ============================================================
summary = {
    "Total Records": len(df_numeric),
    "True Physiological Anomalies": int(sum(y_true)),
    "Best Threshold": best_choice["Threshold"],
    "Best Threshold Value": final_thr,
    "AUC": auc_all,
    "Precision": precision_final,
    "Recall": recall_final,
    "F1": f1_final,
    "FAR": far_final,
    "Predicted Anomalies": int(sum(y_pred_final)),
    "Features Used": ", ".join(df_numeric.columns)
}

summary_df = pd.DataFrame([summary])
summary_file = SUMMARY_DIR / "LOF_DATETIMEEVENTS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"📄 Final summary saved → {summary_file}")

# ============================================================
# Visualization
# ============================================================
plt.figure(figsize=(12, 6))
plt.scatter(np.arange(len(lof_scores)), lof_scores, c=y_pred_final, cmap="coolwarm", s=10)
plt.axhline(final_thr, color="orange", linestyle="--", label=f"Best Threshold ({best_choice['Threshold']})")
plt.title("LOF Anomaly Detection - DATETIMEEVENTS.csv")
plt.xlabel("Index")
plt.ylabel("LOF Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DATETIMEEVENTS_LOF_AnomalyPlot.png")
plt.close()

print("\n🎯 LOF anomaly detection with multi-threshold analysis completed successfully!")
