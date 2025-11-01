import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\IsolationForest-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Thresholds for vital signs (only those present)
# ======================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR"],
    "SBP": ["SysBP", "SBP", "SystolicBP"]
}

# ======================
# Label anomalies based on physiological thresholds
# ======================
def label_anomalies(series, col_name, gt_array, index_offset):
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if series.empty:
        return []
    vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
    anomalous_indices = []

    if vital == "HR":
        anomalous_indices += list(series[series > THRESHOLDS["HR_tachy"]].index)
        anomalous_indices += list(series[series < THRESHOLDS["HR_brady"]].index)
    elif vital == "SBP":
        anomalous_indices += list(series[series > THRESHOLDS["SBP_high"]].index)
        anomalous_indices += list(series[series < THRESHOLDS["SBP_low"]].index)

    for i in anomalous_indices:
        if i + index_offset < len(gt_array):
            gt_array[i + index_offset] = 1

    return [{"vital": col_name, "index": i + index_offset} for i in anomalous_indices]


# ======================
# Evaluation Function
# ======================
def evaluate_anomaly_detection(y_true, y_pred, scores):
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far
    }

# ======================
# Main Processing
# ======================
print(f"\n=== Processing {DATA_PATH.name} using Isolation Forest ===")
df = pd.read_csv(DATA_PATH)
df = df[["SysBP", "Pulse"]]  # Keep relevant vitals only
df = df.dropna().reset_index(drop=True)

# Ground Truth from physiological thresholds
gt_array = np.zeros(len(df))
anomalies = []
for col in df.columns:
    anomalies.extend(label_anomalies(df[col], col, gt_array, 0))

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ======================
# Train Isolation Forest
# ======================
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # expected proportion of anomalies
    random_state=42,
    bootstrap=True
)
iso_forest.fit(X_scaled)

# Predict anomalies (-1 = anomaly, 1 = normal)
y_pred_iso = iso_forest.predict(X_scaled)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # convert to 1 = anomaly, 0 = normal

# Get anomaly scores
scores = -iso_forest.score_samples(X_scaled)

# Evaluate performance
results = evaluate_anomaly_detection(gt_array, y_pred_iso, scores)
print(f"\nResults for Isolation Forest:")
print(f"AUC={results['AUC']:.4f} | F1={results['F1']:.4f} | "
      f"Precision={results['Precision']:.4f} | Recall={results['Recall']:.4f} | FAR={results['FAR']:.4f}")

# ======================
# Summary: Count and Save Anomalies
# ======================
summary_data = []
for vital in df.columns:
    indices = [a["index"] for a in anomalies if a["vital"] == vital]
    summary_data.append({
        "Vital Sign": vital,
        "Total Anomalies (Physiological)": len(indices),
        "Anomaly Record Indices (Physiological)": ", ".join(map(str, indices)) if indices else "None"
    })

# Add Isolation Forest detected anomalies
iso_anomaly_indices = np.where(y_pred_iso == 1)[0]
summary_data.append({
    "Vital Sign": "Model (IsolationForest)",
    "Total Anomalies (Detected)": len(iso_anomaly_indices),
    "Anomaly Record Indices (Detected)": ", ".join(map(str, iso_anomaly_indices)) if len(iso_anomaly_indices) else "None"
})

summary_df = pd.DataFrame(summary_data)
summary_file = SUMMARY_DIR / "IsolationForest_anomaly_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"\n🧾 Anomaly summary saved → {summary_file}")

# ======================
# Plot anomalies
# ======================
plt.figure(figsize=(12,6))
plt.plot(df["SysBP"], label="SysBP", alpha=0.7)
plt.plot(df["Pulse"], label="Pulse", alpha=0.7)

# Mark anomalies
for idx in iso_anomaly_indices:
    plt.scatter(idx, df.loc[idx, "SysBP"], color="red", marker="x", s=50)
    plt.scatter(idx, df.loc[idx, "Pulse"], color="red", marker="x", s=50)

plt.title("ICU Anomaly Detection using Isolation Forest (SysBP & Pulse)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICU_IsolationForest_anomaly_plot.png")
plt.close()

print(f"\n✅ Results and plots saved in → {OUTPUT_DIR}")
