import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import chardet

# ===================== PATHS =====================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\LOF-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for Local Outlier Factor (LOF) Anomaly Detection ===")

# ===================== ENCODING DETECTION =====================
with open(DATA_PATH, "rb") as f:
    rawdata = f.read(4096)
result = chardet.detect(rawdata)
encoding = result["encoding"]
print(f"✅ Detected encoding: {encoding}")

# ===================== DELIMITER DETECTION =====================
with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    sample_lines = [next(f) for _ in range(10)]
sample_text = "\n".join(sample_lines)
delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: sample_text.count(d))
print(f"🔍 Auto-detected best delimiter: '{best_delim}'")

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ File loaded → Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ===================== PREPROCESSING =====================
# Convert time columns
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Create time difference if possible
if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.append("time_diff_min")

df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna(how="any")
if df_numeric.empty:
    raise ValueError("❌ No usable numeric columns found in LABEVENTS.csv for LOF model.")

print(f"🧮 Using features: {numeric_cols}")

# ===================== SCALE DATA =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ===================== TRAIN LOF MODEL =====================
print("\n🚀 Training Local Outlier Factor model...")
lof = LocalOutlierFactor(
    n_neighbors=25,  # typical range 20–50
    contamination=0.05,  # expected % of anomalies
    novelty=False
)
y_pred = lof.fit_predict(X_scaled)

# LOF score (lower = more anomalous)
lof_scores = -lof.negative_outlier_factor_

# ===================== DETECT ANOMALIES =====================
threshold = np.percentile(lof_scores, 95)
anomaly_mask = lof_scores > threshold

print(f"✅ LOF model trained successfully.")
print(f"🚨 Detected {anomaly_mask.sum()} anomalies out of {len(lof_scores)} (threshold={threshold:.4f})")

# ===================== SAVE RESULTS =====================
results_df = pd.DataFrame({
    "index": df_numeric.index,
    "lof_score": lof_scores,
    "is_anomaly": anomaly_mask.astype(int)
})
results_file = OUTPUT_DIR / "LABEVENTS_LOF_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"✅ Results saved → {results_file}")

summary = {
    "total_records": len(lof_scores),
    "detected_anomalies": int(anomaly_mask.sum()),
    "anomaly_percentage": round(100 * anomaly_mask.sum() / len(lof_scores), 3),
    "lof_threshold_95pct": float(threshold),
    "n_neighbors": 25,
    "contamination": 0.05
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "LOF_LABEVENTS_Summary.csv", index=False)
print(f"✅ Summary saved → {SUMMARY_DIR / 'LOF_LABEVENTS_Summary.csv'}")

# ===================== PLOT RESULTS =====================
plt.figure(figsize=(12, 6))
plt.title("Local Outlier Factor (LOF) - LABEVENTS Anomaly Detection")
plt.plot(lof_scores, label="LOF score", alpha=0.7)
plt.axhline(threshold, color="orange", linestyle="--", label="95th percentile threshold")
plt.scatter(np.where(anomaly_mask)[0], lof_scores[anomaly_mask], color="red", marker="x", label="Anomalies")
plt.xlabel("Record Index")
plt.ylabel("LOF Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_LOF_AnomalyPlot.png")
plt.close()
print(f"✅ Plot saved → {OUTPUT_DIR / 'LABEVENTS_LOF_AnomalyPlot.png'}")

print("\n🎯 LOF-based Anomaly Detection completed successfully!")
