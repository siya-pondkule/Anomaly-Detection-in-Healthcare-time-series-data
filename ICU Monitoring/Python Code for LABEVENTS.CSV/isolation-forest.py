import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

# Detect delimiter
with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    sample_lines = "\n".join([next(f) for _ in range(10)])
delim_candidates = [",", ";", "|", "\t"]
delim_counts = {d: sample_lines.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"Detected delimiter: '{best_delim}' (counts={delim_counts})")

# ====================== Load CSV ======================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ Loaded data → shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ====================== Preprocessing ======================
# Parse time columns if exist
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Create time difference feature
if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
elif "charttime" in df.columns:
    df.sort_values("charttime", inplace=True)
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.append("time_diff_min")

print(f"🧮 Using numeric columns: {numeric_cols}")

df_numeric = df[numeric_cols].copy()
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
df_numeric.dropna(how="all", inplace=True)

if df_numeric.empty:
    raise ValueError("❌ No numeric data available for Isolation Forest training.")

# ====================== Scale Features ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ====================== Train Isolation Forest ======================
print("🚀 Training Isolation Forest model...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # 5% anomalies
    max_samples="auto",
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_scaled)

# ====================== Predict Anomalies ======================
scores = iso_forest.decision_function(X_scaled)
predictions = iso_forest.predict(X_scaled)
# In sklearn: -1 = anomaly, 1 = normal
anomaly_mask = predictions == -1

print(f"✅ Model trained successfully!")
print(f"🚨 Detected {anomaly_mask.sum()} anomalies out of {len(predictions)} records.")

# ====================== Save Results ======================
results_df = pd.DataFrame({
    "anomaly_score": scores,
    "is_anomaly": anomaly_mask.astype(int)
})
results_file = OUTPUT_DIR / "LABEVENTS_IsolationForest_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"✅ Results saved → {results_file}")

summary = {
    "total_records": len(predictions),
    "detected_anomalies": int(anomaly_mask.sum()),
    "anomaly_percentage": round(100.0 * anomaly_mask.sum() / len(predictions), 3),
    "contamination": 0.05,
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "IsolationForest_LABEVENTS_Summary.csv", index=False)
print(f"✅ Summary saved → {SUMMARY_DIR / 'IsolationForest_LABEVENTS_Summary.csv'}")

# ====================== Plot Results ======================
plt.figure(figsize=(12,6))
plt.title("Isolation Forest Anomaly Detection - LABEVENTS")
plt.plot(scores, label="Anomaly Score", alpha=0.7)
plt.scatter(np.where(anomaly_mask)[0], scores[anomaly_mask], color="red", marker="x", label="Anomalies")
plt.xlabel("Record Index")
plt.ylabel("Isolation Forest Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_IsolationForest_AnomalyPlot.png")
plt.close()
print(f"✅ Plot saved → {OUTPUT_DIR / 'LABEVENTS_IsolationForest_AnomalyPlot.png'}")

print("\n🎯 Isolation Forest training and anomaly detection completed successfully!")
