import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import csv

# ============================================================
# Paths
# ============================================================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\IsolationForest-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Isolation Forest Anomaly Detection ===")

# ============================================================
# Smart encoding & delimiter detection
# ============================================================
# Detect encoding
try:
    with open(DATA_PATH, 'rb') as f:
        sample = f.read(4096)
    encoding = 'utf-8'
    sample.decode(encoding)
except Exception:
    encoding = 'latin1'
print(f"✅ Detected encoding: {encoding}")

# Detect best delimiter
with open(DATA_PATH, 'r', encoding=encoding, errors='ignore') as f:
    sample = [next(f) for _ in range(10)]
sample_text = "\n".join(sample)
delim_candidates = [',', ';', '|', '\t']
delim_counts = {d: sample_text.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"🔍 Auto-detected best delimiter: '{best_delim}' (counts={delim_counts})")

df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine='python', on_bad_lines='skip')
print(f"✅ File loaded → Shape: {df.shape}")

# ============================================================
# Preprocessing
# ============================================================
if 'charttime' in df.columns:
    df['charttime'] = pd.to_datetime(df['charttime'], errors='coerce')

if 'subject_id' in df.columns and 'charttime' in df.columns:
    df.sort_values(by=['subject_id', 'charttime'], inplace=True)
    df['time_diff_min'] = df.groupby('subject_id')['charttime'].diff().dt.total_seconds() / 60
    df['time_diff_min'] = df['time_diff_min'].fillna(0)

if 'subject_id' in df.columns:
    event_counts = df.groupby('subject_id').size().rename('event_count')
    mean_timediff = df.groupby('subject_id')['time_diff_min'].mean().rename('avg_time_gap_min')
    df_features = pd.concat([event_counts, mean_timediff], axis=1).reset_index()
else:
    df['time_diff_min'] = df['charttime'].diff().dt.total_seconds() / 60
    df['time_diff_min'] = df['time_diff_min'].fillna(0)
    df_features = df[['time_diff_min']].copy()

df_features = df_features.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)
print(f"🧮 Derived features for Isolation Forest: {list(df_features.columns)}")

if df_features.empty:
    raise ValueError("❌ No usable numeric columns found in DATETIMEEVENTS.csv for Isolation Forest training.")

# ============================================================
# Scaling
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# ============================================================
# Train Isolation Forest
# ============================================================
print("🚀 Training Isolation Forest...")
isoforest = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # 5% anomalies expected
    random_state=42
)
isoforest.fit(X_scaled)

# Compute anomaly scores
scores = -isoforest.decision_function(X_scaled)
labels = isoforest.predict(X_scaled)
labels = np.where(labels == -1, "Anomaly", "Normal")

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame({
    "Index": np.arange(len(df_features)),
    "Anomaly_Score": scores,
    "Anomaly_Label": labels
})
results_file = OUTPUT_DIR / "DATETIMEEVENTS_IsolationForest_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"💾 Results saved → {results_file}")

# ============================================================
# Summary
# ============================================================
summary_data = {
    "Total Records": len(df_features),
    "Detected Anomalies": int(np.sum(labels == "Anomaly")),
    "Anomaly Percentage (%)": round(100 * np.mean(labels == "Anomaly"), 2),
    "Features Used": list(df_features.columns)
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "IsolationForest_DATETIMEEVENTS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"📄 Summary saved → {summary_file}")

# ============================================================
# Visualization
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(scores, label="Anomaly Score", color="blue", alpha=0.7)
threshold = np.percentile(scores, 95)
plt.axhline(threshold, color="orange", linestyle="--", label="Threshold (95th %ile)")
plt.scatter(np.where(scores > threshold), scores[scores > threshold], color="red", marker="x", label="Anomalies")
plt.title("Isolation Forest Anomaly Detection - DATETIMEEVENTS.csv")
plt.xlabel("Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DATETIMEEVENTS_IsolationForest_AnomalyPlot.png")
plt.close()

print(f"📊 Plot saved → {OUTPUT_DIR / 'DATETIMEEVENTS_IsolationForest_AnomalyPlot.png'}")
print("\n✅ Isolation Forest training and anomaly detection completed successfully!")
