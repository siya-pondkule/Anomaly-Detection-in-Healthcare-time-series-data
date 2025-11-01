import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from pathlib import Path
import matplotlib.pyplot as plt
import csv

# ========== Paths ==========
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\OneClassSVM-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for One-Class SVM Anomaly Detection ===")

# ========== Encoding & delimiter detection ==========
try:
    with open(DATA_PATH, "rb") as f:
        sample = f.read(4096)
    encoding = "utf-8"
    sample.decode(encoding)
except Exception:
    encoding = "latin1"
print(f"Detected encoding: {encoding}")

# detect delimiter
with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    line = f.readline()
delim = "," if line.count(",") >= line.count(";") else ";"
print(f"Detected delimiter: '{delim}'")

# ========== Load data ==========
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=delim, engine="python", on_bad_lines="skip")
print(f"Loaded shape: {df.shape}")

# Parse charttime if available
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

# Create time_diff feature
if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
elif "charttime" in df.columns:
    df.sort_values("charttime", inplace=True)
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60.0
else:
    df["time_diff_min"] = np.arange(len(df))

df["time_diff_min"] = df["time_diff_min"].fillna(0.0)

# ========== Select numeric columns ==========
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.append("time_diff_min")
print(f"Using numeric columns for model: {numeric_cols}")

df_numeric = df[numeric_cols].copy()

# Drop NaN and Inf safely
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
df_numeric.dropna(inplace=True)

if df_numeric.empty:
    raise ValueError("❌ No valid numeric rows after cleaning. Check dataset preprocessing!")

print(f"Remaining numeric data shape after cleaning: {df_numeric.shape}")

# ========== Standardize ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)
print("✅ Data scaled successfully.")

# ========== One-Class SVM ==========
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
svm.fit(X_scaled)
y_pred = svm.predict(X_scaled)
y_score = svm.decision_function(X_scaled)

# Map predictions: -1 → anomaly, 1 → normal
is_anomaly = (y_pred == -1).astype(int)
anomaly_count = is_anomaly.sum()

print(f"Detected {anomaly_count} anomalies out of {len(y_pred)} samples")

# ========== Save results ==========
results_df = pd.DataFrame({
    "index": np.arange(len(df_numeric)),
    "decision_score": y_score,
    "is_anomaly": is_anomaly
})
results_file = OUTPUT_DIR / "DATETIMEEVENTS_OneClassSVM_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# Summary file
summary = {
    "total_samples": len(y_pred),
    "detected_anomalies": int(anomaly_count),
    "anomaly_percentage": round(100 * anomaly_count / len(y_pred), 2)
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "OneClassSVM_DATETIMEEVENTS_Summary.csv", index=False)
print(f"Summary saved → {SUMMARY_DIR / 'OneClassSVM_DATETIMEEVENTS_Summary.csv'}")

# ========== Plot ==========
plt.figure(figsize=(10, 6))
plt.plot(y_score, label="Decision function score")
plt.scatter(np.where(is_anomaly)[0], y_score[is_anomaly == 1], color="red", marker="x", label="Anomalies")
plt.xlabel("Sample index")
plt.ylabel("Decision score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DATETIMEEVENTS_OneClassSVM_AnomalyPlot.png")
plt.close()
print(f"Plot saved → {OUTPUT_DIR / 'DATETIMEEVENTS_OneClassSVM_AnomalyPlot.png'}")

print("\n✅ One-Class SVM anomaly detection completed successfully.")
