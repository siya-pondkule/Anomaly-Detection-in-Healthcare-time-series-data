import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

print("\n=== Training LOF anomaly detection model on ICU.csv ===")

# =========================
# 1️⃣ Setup paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\LOF-ModelResult"
os.makedirs(results_dir, exist_ok=True)

# =========================
# 2️⃣ Load dataset
# =========================
df = pd.read_csv(input_path)

# Select vital signs
features = ['SysBP', 'Pulse']  
df = df[features].dropna().reset_index(drop=True)

# =========================
# 3️⃣ Physiological Ground Truth
# =========================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

gt = np.zeros(len(df))

# HR anomalies
gt[df['Pulse'] > THRESHOLDS["HR_tachy"]] = 1
gt[df['Pulse'] < THRESHOLDS["HR_brady"]] = 1

# SBP anomalies
gt[df['SysBP'] > THRESHOLDS["SBP_high"]] = 1
gt[df['SysBP'] < THRESHOLDS["SBP_low"]] = 1

# =========================
# 4️⃣ Preprocessing
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================
# 5️⃣ Train LOF model
# =========================
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=False
)

y_pred = lof.fit_predict(X_scaled)
y_pred = np.where(y_pred == -1, 1, 0)  # 1 = anomaly

# LOF anomaly score (higher = more anomalous)
scores = -lof.negative_outlier_factor_

# =========================
# 6️⃣ Evaluation Metrics
# =========================
try:
    auc = roc_auc_score(gt, scores)
except:
    auc = np.nan

precision = precision_score(gt, y_pred, zero_division=0)
recall = recall_score(gt, y_pred, zero_division=0)
f1 = f1_score(gt, y_pred, zero_division=0)

cm = confusion_matrix(gt, y_pred)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
far = fp / (fp + tn) if (fp + tn) > 0 else 0

# =========================
# 7️⃣ Create Summary File
# =========================
summary_df = pd.DataFrame([{
    "Model": "Local Outlier Factor (LOF)",
    "AUC": auc,
    "F1-score": f1,
    "Precision": precision,
    "Recall": recall,
    "FAR": far
}])

summary_csv = os.path.join(results_dir, "LOF_Anomaly_Summary.csv")
summary_df.to_csv(summary_csv, index=False)

print("\n=== LOF Summary ===")
print(summary_df)

# =========================
# 8️⃣ Save anomaly plot
# =========================
plot_path = os.path.join(results_dir, "LOF_Anomaly_Plot.png")

plt.figure(figsize=(8,6))
plt.scatter(df['SysBP'], df['Pulse'], c=y_pred, cmap='coolwarm')
plt.xlabel("SysBP")
plt.ylabel("Pulse")
plt.title("LOF Anomaly Detection on ICU Data")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 9️⃣ Final message
# =========================
print(f"\n✅ Results saved successfully in: {results_dir}")
print(f"   - Summary CSV: {summary_csv}")
print(f"   - Plot: {plot_path}")
