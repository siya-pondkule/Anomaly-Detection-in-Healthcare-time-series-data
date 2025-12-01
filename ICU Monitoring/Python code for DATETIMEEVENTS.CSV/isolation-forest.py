import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

# =========================
# Paths (modify if needed)
# =========================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\IsolationForest-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for Isolation Forest Anomaly Detection (with full summary metrics) ===")

# =========================
# Smart encoding & delimiter detection (robust load)
# =========================
# Detect encoding
try:
    with open(DATA_PATH, 'rb') as f:
        sample = f.read(4096)
    encoding = 'utf-8'
    sample.decode(encoding)
except Exception:
    encoding = 'latin1'
print(f"✅ Detected encoding: {encoding}")

# Detect delimiter by sampling first lines
with open(DATA_PATH, 'r', encoding=encoding, errors='ignore') as f:
    # safe sample of up to 10 lines
    sample_lines = []
    for _ in range(10):
        try:
            sample_lines.append(next(f))
        except StopIteration:
            break

sample_text = "\n".join(sample_lines)
delim_candidates = [',', ';', '|', '\t']
delim_counts = {d: sample_text.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"🔍 Auto-detected best delimiter: '{best_delim}' (counts={delim_counts})")

# Load CSV robustly (skip bad lines)
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine='python', on_bad_lines='skip')
print(f"✅ File loaded → Shape: {df.shape}")

# =========================
# Feature derivation (same logic used previously)
# =========================
if 'charttime' in df.columns:
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')

if 'subject_id' in df.columns and 'charttime' in df.columns:
    df.sort_values(by=['subject_id', 'charttime'], inplace=True)
    df['time_diff_min'] = df.groupby('subject_id')['charttime'].diff().dt.total_seconds() / 60
    df['time_diff_min'] = df['time_diff_min'].fillna(0)

# Derive per-subject features if possible
if 'subject_id' in df.columns:
    event_counts = df.groupby('subject_id').size().rename('event_count')
    mean_timediff = df.groupby('subject_id')['time_diff_min'].mean().rename('avg_time_gap_min')
    # reset_index(drop=True) used previously — keep the same behavior
    df_features = pd.concat([event_counts, mean_timediff], axis=1).reset_index(drop=True)
else:
    # fallback: use time_diff_min as the single feature
    if 'time_diff_min' not in df.columns:
        # try to compute a simple time diff even if charttime missing (safe fallback)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No usable numeric columns found in the dataset to derive features.")
        df_features = df[numeric_cols].copy().reset_index(drop=True)
    else:
        df_features = df[['time_diff_min']].copy().reset_index(drop=True)

# keep only numeric features
df_features = df_features.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)
print(f"🧮 Derived numeric features for Isolation Forest: {list(df_features.columns)} (shape={df_features.shape})")

if df_features.empty:
    raise ValueError("❌ No usable numeric columns found in DATETIMEEVENTS.csv for Isolation Forest training.")

# =========================
# Build a ground-truth label (Option A style / heuristic)
# - If the feature is time_diff_min or avg_time_gap_min, mark unusually large gaps as anomalies
# - If the feature is event_count, mark unusually high counts as anomalies
# - This provides y_true needed to compute AUC/F1/etc.
# NOTE: This is a heuristic labelling strategy required because DATETIMEEVENTS lacks clinical labels.
# =========================
y_true = np.zeros(len(df_features), dtype=int)

# For each numeric column, mark outliers using robust threshold: > mean + 3*std (or < mean - 3*std for symmetric)
for col in df_features.columns:
    col_vals = df_features[col].values
    mu = np.nanmean(col_vals)
    sigma = np.nanstd(col_vals)
    if sigma == 0 or np.isnan(sigma):
        continue
    # treat directional anomalies appropriately
    if 'time' in col.lower() or 'gap' in col.lower() or 'avg' in col.lower():
        # long gaps often signify unusual behavior → mark high values
        anomalous_mask = col_vals > (mu + 3 * sigma)
    else:
        # event_count or other numeric: mark both unusually high and unusually low as anomalies
        anomalous_mask = (col_vals > (mu + 3 * sigma)) | (col_vals < (mu - 3 * sigma))
    y_true = np.where(anomalous_mask, 1, y_true)  # once labelled anomaly remain 1

num_true_anoms = int(np.sum(y_true))
print(f"⚠️ Heuristic ground-truth anomalies (derived): {num_true_anoms} / {len(y_true)} records")

# If no ground truth anomalies found by heuristic, fall back to percentile-based rule on first numeric column
if num_true_anoms == 0:
    col = df_features.columns[0]
    vals = df_features[col].values
    perc_high = np.percentile(vals, 99)
    y_true = (vals >= perc_high).astype(int)
    num_true_anoms = int(np.sum(y_true))
    print(f"ℹ️ Fallback GT applied on column '{col}' using 99th percentile -> anomalies: {num_true_anoms}")

# =========================
# Scaling
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# =========================
# Train Isolation Forest
# =========================
isof = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
isof.fit(X_scaled)

# Anomaly scores (higher = more anomalous). Use negative of decision_function so higher = more abnormal
scores = -isof.decision_function(X_scaled)

# Default model predictions (using model internal threshold)
pred_model = isof.predict(X_scaled)  # -1 anomaly, 1 normal
y_pred_model = np.where(pred_model == -1, 1, 0)

# =========================
# MULTI-THRESHOLD EVALUATION (REPLACING the original sweep)
# Evaluate only at 90%, 95%, 98% percentiles of the 'scores'
# Compute AUC, Precision, Recall, F1, FAR for each threshold. Select best by F1.
# =========================
threshold_percentiles = {
    "90%": 90,
    "95%": 95,
    "98%": 98
}

multi_results = []
best_by_f1 = None

# Pre-compute AUC using continuous scores if possible
try:
    auc_all = roc_auc_score(y_true, scores) if (np.unique(y_true).size > 1) else np.nan
except Exception:
    auc_all = np.nan

for name, pct in threshold_percentiles.items():
    thr_value = np.percentile(scores, pct)
    y_pred_thr = (scores >= thr_value).astype(int)

    # compute metrics
    # AUC is same continuous score AUC for all thresholds (but saved per-threshold for completeness)
    auc_thr = auc_all
    precision_thr = precision_score(y_true, y_pred_thr, zero_division=0)
    recall_thr = recall_score(y_true, y_pred_thr, zero_division=0)
    f1_thr = f1_score(y_true, y_pred_thr, zero_division=0)

    # FAR: FP / (FP + TN)
    cm_thr = confusion_matrix(y_true, y_pred_thr)
    if cm_thr.size == 4:
        tn, fp, fn, tp = cm_thr.ravel()
    else:
        # degenerate case (all one class)
        tn = cm_thr[0, 0] if cm_thr.size == 1 else 0
        fp = fn = tp = 0
    far_thr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    res = {
        "Threshold_Name": name,
        "Percentile": pct,
        "Threshold_Value": float(thr_value),
        "AUC": float(auc_thr) if not np.isnan(auc_thr) else np.nan,
        "Precision": float(precision_thr),
        "Recall": float(recall_thr),
        "F1": float(f1_thr),
        "FAR": float(far_thr),
        "Predicted_Anomalies": int(y_pred_thr.sum())
    }
    multi_results.append(res)

    # choose best by F1, tie-breaker higher precision
    if best_by_f1 is None:
        best_by_f1 = res
    else:
        if res["F1"] > best_by_f1["F1"]:
            best_by_f1 = res
        elif res["F1"] == best_by_f1["F1"]:
            # tie break: choose higher precision
            if res["Precision"] > best_by_f1["Precision"]:
                best_by_f1 = res

# Print multi-threshold table
print("\n=== Multi-threshold evaluation (90,95,98 percentiles) ===")
for r in multi_results:
    print(r)

print("\n=== Best threshold among [90,95,98] (selected by highest F1, tie -> precision) ===")
print(best_by_f1)

# Save threshold summary CSV
threshold_summary_df = pd.DataFrame(multi_results)
threshold_summary_file = SUMMARY_DIR / "IsolationForest_DATETIMEEVENTS_MultiThreshold_Summary.csv"
threshold_summary_df.to_csv(threshold_summary_file, index=False)
print(f"\nMulti-threshold summary saved → {threshold_summary_file}")

# =========================
# Use the chosen best threshold to compute final predictions and metrics
# =========================
if best_by_f1 and best_by_f1.get("Threshold_Value") is not None:
    final_threshold = best_by_f1["Threshold_Value"]
    y_pred_final = (scores >= final_threshold).astype(int)
else:
    # fallback to model's internal prediction if best threshold not found
    y_pred_final = y_pred_model
    final_threshold = None

# final metrics based on chosen threshold
try:
    auc_final = float(roc_auc_score(y_true, scores)) if (np.unique(y_true).size > 1) else np.nan
except Exception:
    auc_final = np.nan

precision_final = float(precision_score(y_true, y_pred_final, zero_division=0))
recall_final = float(recall_score(y_true, y_pred_final, zero_division=0))
f1_final = float(f1_score(y_true, y_pred_final, zero_division=0))
cm_final = confusion_matrix(y_true, y_pred_final)
if cm_final.size == 4:
    tn, fp, fn, tp = cm_final.ravel()
else:
    tn = cm_final[0, 0] if cm_final.size == 1 else 0
    fp = fn = tp = 0
far_final = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

print("\n=== Final evaluation (using chosen best threshold) ===")
print(f"AUC: {auc_final}")
print(f"Precision: {precision_final}")
print(f"Recall: {recall_final}")
print(f"F1: {f1_final}")
print(f"FAR: {far_final}")
print(f"Chosen Threshold Value: {final_threshold} (from {best_by_f1['Threshold_Name'] if best_by_f1 else 'N/A'})")
print(f"Predicted anomalies at chosen threshold: {int(y_pred_final.sum())}")

# =========================
# Save results (per-record) and summary (single CSV) with required fields
# =========================
results_df = pd.DataFrame({
    "Index": np.arange(len(df_features)),
    "Anomaly_Score": scores,
    "Model_Prediction_Internal": y_pred_model,
    "Predicted_Anomaly_At_Chosen_Threshold": y_pred_final,
    "GT_Anomaly": y_true
})
results_file = OUTPUT_DIR / "DATETIMEEVENTS_IsolationForest_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"\nResults saved → {results_file}")

summary_data = {
    "Total Records": int(len(df_features)),
    "GT_Anomalies_Derived": int(num_true_anoms),
    "Chosen_Threshold_Name": best_by_f1["Threshold_Name"] if best_by_f1 else "",
    "Chosen_Threshold_Value": float(best_by_f1["Threshold_Value"]) if best_by_f1 and best_by_f1.get("Threshold_Value") is not None else "",
    "AUC": float(auc_final) if not np.isnan(auc_final) else "",
    "Precision": float(precision_final),
    "Recall": float(recall_final),
    "F1": float(f1_final),
    "FAR": float(far_final),
    "Predicted_Anomalies_at_Chosen_Threshold": int(y_pred_final.sum()),
    "Features_Used": ";".join(list(df_features.columns))
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "IsolationForest_DATETIMEEVENTS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved → {summary_file}")

# =========================
# Save a simple plot of scores + chosen threshold
# =========================
plt.figure(figsize=(12, 5))
plt.plot(scores, label="Anomaly Score (higher = more anomalous)")
if final_threshold is not None:
    plt.axhline(final_threshold, color="orange", linestyle="--", label=f"Chosen Threshold ({final_threshold:.6f})")
plt.scatter(np.where(y_true == 1), scores[y_true == 1], color='red', marker='x', label='Derived GT anomalies')
plt.title("Isolation Forest - DATETIMEEVENTS Anomaly Scores")
plt.xlabel("Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_file = OUTPUT_DIR / "DATETIMEEVENTS_IsolationForest_ScoresPlot.png"
plt.savefig(plot_file, dpi=300)
plt.close()
print(f"Plot saved → {plot_file}")

print("\n✅ Isolation Forest processing completed with multi-threshold summary and chosen best threshold.")
