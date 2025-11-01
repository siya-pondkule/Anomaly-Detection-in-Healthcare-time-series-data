import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\LOF-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Local Outlier Factor Anomaly Detection ===")

# ======================
# Load and preprocess data
# ======================
df = pd.read_csv(DATA_PATH)

# Keep only numeric columns and remove ID/time/date/unit columns
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for LOF.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# ======================
# Scale features
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================
# Train Local Outlier Factor
# ======================
print("\nTraining Local Outlier Factor model...")
lof = LocalOutlierFactor(
    n_neighbors=20,     # number of neighbors to compare each point
    contamination=0.05, # expected fraction of anomalies (adjustable)
    metric='euclidean', 
    novelty=False       # unsupervised mode (no separate predict step)
)

y_pred = lof.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
anomalies = np.where(y_pred == -1)[0]
scores = -lof.negative_outlier_factor_  # higher scores = more anomalous

print(f"\nDetected {len(anomalies)} anomalies out of {len(X_scaled)} records.")

# ======================
# Compute threshold
# ======================
threshold = np.percentile(scores, 95)

# ======================
# Save results
# ======================
results_df = pd.DataFrame({
    "Index": np.arange(len(X_scaled)),
    "LOF_Score": scores,
    "Anomaly_Label": np.where(y_pred == -1, "Anomaly", "Normal")
})
results_file = OUTPUT_DIR / "ICUSTAYS_LOF_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# ======================
# Summary
# ======================
summary_data = {
    "Total Records": len(X_scaled),
    "Detected Anomalies": len(anomalies),
    "Anomaly Percentage (%)": round(100 * len(anomalies) / len(X_scaled), 2),
    "Neighbors": 20,
    "Contamination": 0.05
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "LOF_ICUSTAYS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved → {summary_file}")

# ======================
# Visualization
# ======================
plt.figure(figsize=(12, 6))
plt.plot(scores, label="LOF Anomaly Score", color="blue", alpha=0.7)
plt.axhline(threshold, color="orange", linestyle="--", label="Anomaly Threshold (95th %ile)")
plt.scatter(anomalies, scores[anomalies], color="red", marker="x", label="Anomalies")
plt.title("Local Outlier Factor (LOF) Anomaly Detection - ICUSTAYS.csv")
plt.xlabel("Index")
plt.ylabel("LOF Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_LOF_AnomalyPlot.png")
plt.close()

print(f"Plot saved → {OUTPUT_DIR / 'ICUSTAYS_LOF_AnomalyPlot.png'}")

print("\n✅ Local Outlier Factor (LOF) anomaly detection complete!")
