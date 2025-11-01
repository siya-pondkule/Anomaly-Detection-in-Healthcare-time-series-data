import pandas as pd
import numpy as np
import chardet
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
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
print(f"✅ Detected encoding: {encoding}")

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    sample = [next(f) for _ in range(10)]
sample_text = "\n".join(sample)
delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: sample_text.count(d))
print(f"🔍 Detected delimiter: '{best_delim}'")

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ Loaded dataset shape: {df.shape}")

# ===================== PREPROCESSING =====================
# Parse timestamps if available
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Create time difference (minutes) feature if possible
if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
else:
    df["time_diff_min"] = np.arange(len(df))

# Choose numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    raise ValueError("❌ No numeric columns found in dataset!")

print(f"🔢 Using numeric columns: {numeric_cols}")

# Clean numeric data
df_numeric = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
if df_numeric.empty:
    raise ValueError("❌ No valid numeric rows after cleaning. Check dataset preprocessing!")

# ===================== SCALING =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)
print(f"✅ Data scaled successfully: {X_scaled.shape}")

# ===================== TRAIN ONE-CLASS SVM =====================
print("\n🧠 Training One-Class SVM...")
ocsvm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)  # nu ≈ expected anomaly fraction
ocsvm.fit(X_scaled)

# ===================== PREDICTIONS =====================
y_pred = ocsvm.predict(X_scaled)
anomaly_scores = ocsvm.decision_function(X_scaled)

# Map predictions: -1 = anomaly, 1 = normal
df_numeric["anomaly_flag"] = (y_pred == -1).astype(int)
df_numeric["anomaly_score"] = -anomaly_scores  # higher = more anomalous

n_anomalies = df_numeric["anomaly_flag"].sum()
print(f"🚨 Detected {n_anomalies} anomalies out of {len(df_numeric)} records ({100*n_anomalies/len(df_numeric):.2f}%)")

# ===================== SAVE RESULTS =====================
results_file = OUTPUT_DIR / "LABEVENTS_OneClassSVM_Results.csv"
df_numeric.to_csv(results_file, index=False)
print(f"✅ Results saved → {results_file}")

summary = {
    "total_records": len(df_numeric),
    "anomalies_detected": int(n_anomalies),
    "anomaly_percentage": round(100 * n_anomalies / len(df_numeric), 2),
    "kernel": "rbf",
    "nu": 0.05
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "OneClassSVM_LABEVENTS_Summary.csv", index=False)
print(f"✅ Summary saved → {SUMMARY_DIR / 'OneClassSVM_LABEVENTS_Summary.csv'}")

# ===================== VISUALIZATION =====================
plt.figure(figsize=(12,6))
plt.plot(df_numeric["anomaly_score"], label="Anomaly Score", alpha=0.7)
plt.scatter(df_numeric.index[df_numeric["anomaly_flag"] == 1],
            df_numeric["anomaly_score"][df_numeric["anomaly_flag"] == 1],
            color="red", label="Anomalies", marker="x")
plt.xlabel("Record Index")
plt.ylabel("Anomaly Score")
plt.title("One-Class SVM Anomaly Detection (LABEVENTS)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_OneClassSVM_AnomalyPlot.png")
plt.close()
print(f"✅ Plot saved → {OUTPUT_DIR / 'LABEVENTS_OneClassSVM_AnomalyPlot.png'}")

print("\n🎯 One-Class SVM Anomaly Detection completed successfully!")
