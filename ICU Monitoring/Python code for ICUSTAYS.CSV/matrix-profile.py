import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import stumpy  # library for matrix profile calculation

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\Matrix-Profile-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)


print(f"\n=== Processing {DATA_PATH.name} for Matrix Profile Anomaly Detection ===")

# ======================
# Load and preprocess data
# ======================
df = pd.read_csv(DATA_PATH)

# Keep numeric and relevant columns only
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No numeric columns found in ICUSTAYS.csv for Matrix Profile analysis.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# ======================
# Scale data
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# If multivariate, take the mean signal across features (simplified univariate anomaly detection)
signal = np.mean(X_scaled, axis=1)

# ======================
# Compute Matrix Profile
# ======================
window_size = 30  # adjust depending on ICU data sampling rate
print(f"\nComputing Matrix Profile with window size = {window_size} ...")

mp = stumpy.stump(signal, m=window_size)
matrix_profile = mp[:, 0]  # the matrix profile values

# ======================
# Detect anomalies
# ======================
# High matrix profile values = subsequences dissimilar to others (potential anomalies)
threshold = np.percentile(matrix_profile, 95)
anomalies = np.where(matrix_profile > threshold)[0]

print(f"\nDetected {len(anomalies)} anomalies out of {len(matrix_profile)} subsequences.")
print(f"Anomaly threshold: {threshold:.6f}")

# ======================
# Save results
# ======================
results_df = pd.DataFrame({
    "Index": np.arange(len(matrix_profile)),
    "Matrix_Profile_Value": matrix_profile,
    "Anomaly_Label": np.where(matrix_profile > threshold, "Anomaly", "Normal")
})
results_file = OUTPUT_DIR / "ICUSTAYS_MatrixProfile_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# ======================
# Summary
# ======================
summary_data = {
    "Total Points": len(matrix_profile),
    "Detected Anomalies": len(anomalies),
    "Anomaly Percentage (%)": round(100 * len(anomalies) / len(matrix_profile), 2)
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "MatrixProfile_ICUSTAYS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved → {summary_file}")

# ======================
# Visualization
# ======================
plt.figure(figsize=(12, 6))
plt.plot(signal, label="ICU Signal (Mean of features)", color="blue", alpha=0.7)
plt.scatter(anomalies, signal[anomalies], color="red", label="Detected Anomalies", marker="x")
plt.title("Matrix Profile Anomaly Detection - ICUSTAYS.csv")
plt.xlabel("Index")
plt.ylabel("Scaled Signal Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_MatrixProfile_AnomalyPlot.png")
plt.close()

# Plot Matrix Profile itself
plt.figure(figsize=(12, 5))
plt.plot(matrix_profile, color="green", label="Matrix Profile", alpha=0.8)
plt.axhline(threshold, color="orange", linestyle="--", label="Anomaly Threshold")
plt.scatter(anomalies, matrix_profile[anomalies], color="red", marker="x", label="Anomalies")
plt.title("Matrix Profile (ICUSTAYS.csv)")
plt.xlabel("Index")
plt.ylabel("Matrix Profile Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_MatrixProfile_Plot.png")
plt.close()

print(f"Plots saved → {OUTPUT_DIR}")
print("\n✅ Matrix Profile anomaly detection complete!")
