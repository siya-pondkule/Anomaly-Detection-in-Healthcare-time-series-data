import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\One-class-SVM-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)


print(f"\n=== Processing {DATA_PATH.name} for One-Class SVM Anomaly Detection ===")

# ======================
# Load and preprocess data
# ======================
df = pd.read_csv(DATA_PATH)

# Keep only numeric columns and remove ID/time/date/unit columns
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for One-Class SVM.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# ======================
# Scale features
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================
# Train One-Class SVM
# ======================
print("\nTraining One-Class SVM...")
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)  # nu = expected anomaly fraction
ocsvm.fit(X_scaled)

# ======================
# Predictions
# ======================
y_pred = ocsvm.predict(X_scaled)
# One-Class SVM labels: -1 = anomaly, 1 = normal
anomalies = np.where(y_pred == -1)[0]

print(f"\nDetected {len(anomalies)} anomalies out of {len(X_scaled)} records.")

# ======================
# Evaluation (unsupervised - no true labels)
# ======================
# We'll compute reconstruction-free metrics like anomaly rate & summary stats
anomaly_rate = 100 * len(anomalies) / len(X_scaled)

# Distance from decision boundary
decision_scores = ocsvm.decision_function(X_scaled)
threshold = np.percentile(decision_scores, 5)

# ======================
# Save results
# ======================
results_df = pd.DataFrame({
    "Index": np.arange(len(X_scaled)),
    "Decision_Score": decision_scores,
    "Anomaly_Label": np.where(y_pred == -1, "Anomaly", "Normal")
})
results_file = OUTPUT_DIR / "ICUSTAYS_OneClassSVM_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# ======================
# Summary
# ======================
summary_data = {
    "Total Records": len(X_scaled),
    "Detected Anomalies": len(anomalies),
    "Anomaly Percentage (%)": round(anomaly_rate, 2),
    "Kernel": "RBF",
    "Nu (Expected Outlier Fraction)": 0.05
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "OneClassSVM_ICUSTAYS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved → {summary_file}")

# ======================
# Visualization
# ======================
plt.figure(figsize=(12, 6))
plt.plot(decision_scores, label="SVM Decision Function", color="blue", alpha=0.7)
plt.axhline(threshold, color="orange", linestyle="--", label="Anomaly Threshold")
plt.scatter(anomalies, decision_scores[anomalies], color="red", marker="x", label="Anomalies")
plt.title("One-Class SVM Anomaly Detection - ICUSTAYS.csv")
plt.xlabel("Index")
plt.ylabel("Decision Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_OneClassSVM_AnomalyPlot.png")
plt.close()

print(f"Plot saved → {OUTPUT_DIR / 'ICUSTAYS_OneClassSVM_AnomalyPlot.png'}")

print("\n✅ One-Class SVM anomaly detection complete!")
