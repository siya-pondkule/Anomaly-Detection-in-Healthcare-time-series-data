import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\Isolation-Forest-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Load and preprocess data
# ======================
print(f"\n=== Processing {DATA_PATH.name} for Isolation Forest ===")
df = pd.read_csv(DATA_PATH)

# Drop non-numeric and irrelevant columns
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["time", "date", "id", "unit"])]
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
numeric_df = numeric_df.dropna().reset_index(drop=True)

if numeric_df.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for Isolation Forest anomaly detection.")

print(f"Using columns for anomaly detection: {list(numeric_df.columns)}")

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

# ======================
# Train Isolation Forest
# ======================
print("\nTraining Isolation Forest model...")
iso_forest = IsolationForest(
    n_estimators=200, 
    contamination=0.05,   # 5% anomalies (tune this if needed)
    random_state=42,
    max_samples='auto',
    n_jobs=-1
)
iso_forest.fit(X_scaled)

# Predict anomalies
predictions = iso_forest.predict(X_scaled)
# -1 = anomaly, 1 = normal
anomaly_indices = np.where(predictions == -1)[0]
scores = -iso_forest.decision_function(X_scaled)  # higher = more anomalous

print(f"Detected {len(anomaly_indices)} anomalies out of {len(predictions)} total records.")

# ======================
# Save anomaly results
# ======================
df_results = numeric_df.copy()
df_results["Anomaly_Score"] = scores
df_results["Anomaly_Label"] = np.where(predictions == -1, "Anomaly", "Normal")

results_file = OUTPUT_DIR / "ICUSTAYS_IsolationForest_Results.csv"
df_results.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# ======================
# Anomaly summary
# ======================
summary_data = {
    "Total Records": len(df_results),
    "Detected Anomalies": len(anomaly_indices),
    "Anomaly Percentage (%)": round(100 * len(anomaly_indices) / len(df_results), 2)
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "IsolationForest_ICUSTAYS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved → {summary_file}")

# ======================
# Plot anomalies
# ======================
plt.figure(figsize=(12, 6))
plt.title("Isolation Forest Anomaly Detection - ICUSTAYS.csv")
plt.plot(scores, label="Anomaly Score", color="blue", alpha=0.7)
plt.scatter(anomaly_indices, scores[anomaly_indices], color="red", label="Detected Anomalies", marker="x")
plt.xlabel("Record Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_IsolationForest_Plot.png")
plt.close()

print(f"Plot saved → {OUTPUT_DIR / 'ICUSTAYS_IsolationForest_Plot.png'}")

print("\n✅ Isolation Forest anomaly detection complete!")
