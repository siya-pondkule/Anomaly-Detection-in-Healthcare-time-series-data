import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
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
# Detect encoding and delimiter
# ============================================================
try:
    with open(DATA_PATH, "rb") as f:
        sample = f.read(4096)
    encoding = "utf-8"
    sample.decode(encoding)
except Exception:
    encoding = "latin1"
print(f"✅ Detected encoding: {encoding}")

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    lines = [next(f) for _ in range(5)]
sample_text = "\n".join(lines)
delim_candidates = [",", ";", "|", "\t"]
delim_counts = {d: sample_text.count(d) for d in delim_candidates}
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
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0)
elif "charttime" in df.columns:
    df.sort_values("charttime", inplace=True)
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

# ============================================================
# Numeric feature selection
# ============================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    raise ValueError("❌ No numeric columns found for LOF training.")

df_numeric = df[numeric_cols].copy()

# ============================================================
# Handle missing and infinite values
# ============================================================
# Replace inf/-inf with NaN
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace NaN with column mean
df_numeric.fillna(df_numeric.mean(), inplace=True)

# If still NaN (e.g. column entirely NaN), fill with 0
df_numeric.fillna(0, inplace=True)

# Drop constant columns (e.g., all values same)
df_numeric = df_numeric.loc[:, df_numeric.apply(pd.Series.nunique) > 1]

# Final check
if df_numeric.isna().any().any():
    raise ValueError("⚠️ Still contains NaN after cleaning!")

print(f"🧮 Using features: {df_numeric.columns.tolist()}")

# ============================================================
# Scaling
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ============================================================
# Train LOF
# ============================================================
print("\n🚀 Training Local Outlier Factor model...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred = lof.fit_predict(X_scaled)

labels = np.where(y_pred == -1, "Anomaly", "Normal")
lof_scores = -lof.negative_outlier_factor_

# ============================================================
# Save results
# ============================================================
results_df = pd.DataFrame({
    "Index": np.arange(len(df_numeric)),
    "LOF_Score": lof_scores,
    "Anomaly_Label": labels
})
results_file = OUTPUT_DIR / "DATETIMEEVENTS_LOF_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"✅ Results saved → {results_file}")

# ============================================================
# Save summary
# ============================================================
summary = {
    "Total Records": len(df_numeric),
    "Detected Anomalies": np.sum(labels == "Anomaly"),
    "Anomaly %": round(100 * np.mean(labels == "Anomaly"), 2),
    "Used Features": ", ".join(df_numeric.columns),
    "Contamination": 0.05,
    "Neighbors": 20
}
summary_df = pd.DataFrame([summary])
summary_file = SUMMARY_DIR / "LOF_DATETIMEEVENTS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"✅ Summary saved → {summary_file}")

# ============================================================
# Visualization
# ============================================================
plt.figure(figsize=(12, 6))
plt.scatter(np.arange(len(lof_scores)), lof_scores, c=(labels == "Anomaly"), cmap="coolwarm", s=8)
plt.axhline(np.percentile(lof_scores, 95), color="orange", linestyle="--", label="Anomaly threshold")
plt.title("Local Outlier Factor - DATETIMEEVENTS.csv")
plt.xlabel("Index")
plt.ylabel("LOF Score (higher = more anomalous)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DATETIMEEVENTS_LOF_AnomalyPlot.png")
plt.close()

print(f"✅ Plot saved → {OUTPUT_DIR / 'DATETIMEEVENTS_LOF_AnomalyPlot.png'}")

print("\n🎯 LOF anomaly detection completed successfully!")
