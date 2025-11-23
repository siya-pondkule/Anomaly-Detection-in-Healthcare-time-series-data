import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

print("\n=== Training One-Class SVM anomaly detection model on ICU.csv ===")

# =========================
# 1️⃣ Setup paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\One-Class-SVM"
os.makedirs(results_dir, exist_ok=True)

# =========================
# 2️⃣ Load and preprocess data
# =========================
df = pd.read_csv(input_path)

# Use same columns as other classical models
features = ['Age', 'SysBP', 'Pulse']
df = df[features].dropna().reset_index(drop=True)

# Physiological thresholds (for Ground Truth)
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

# Create ground truth array
gt_array = np.zeros(len(df))

# SBP rules
gt_array[df["SysBP"] > THRESHOLDS["SBP_high"]] = 1
gt_array[df["SysBP"] < THRESHOLDS["SBP_low"]] = 1

# Pulse rules
gt_array[df["Pulse"] > THRESHOLDS["HR_tachy"]] = 1
gt_array[df["Pulse"] < THRESHOLDS["HR_brady"]] = 1

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================
# 3️⃣ Train One-Class SVM
# =========================
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
y_pred = svm.fit_predict(X_scaled)

# Convert prediction: -1 = anomaly, 1 = normal
y_pred = np.where(y_pred == -1, 1, 0)

# =========================
# 4️⃣ Calculate Evaluation Metrics
# =========================
scores = -svm.score_samples(X_scaled)

try:
    auc = roc_auc_score(gt_array, scores)
except:
    auc = np.nan

precision = precision_score(gt_array, y_pred, zero_division=0)
recall = recall_score(gt_array, y_pred, zero_division=0)
f1 = f1_score(gt_array, y_pred, zero_division=0)

cm = confusion_matrix(gt_array, y_pred)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
far = fp / (fp + tn) if (fp + tn) else 0

# =========================
# 5️⃣ Summary (UPDATED)
# =========================
summary_text = (
    f"=== One-Class SVM Anomaly Detection Summary ===\n"
    f"Total samples: {len(df)}\n"
    f"Detected anomalies: {y_pred.sum()}\n\n"
    f"--- Evaluation Metrics ---\n"
    f"AUC Score: {auc:.4f}\n"
    f"Precision: {precision:.4f}\n"
    f"Recall: {recall:.4f}\n"
    f"F1-score: {f1:.4f}\n"
    f"FAR (False Alarm Rate): {far:.4f}\n"
)

print("\n" + summary_text)

# =========================
# 6️⃣ Save results
# =========================
df_results = df.copy()
df_results["Anomaly"] = y_pred
df_results["Anomaly_Score"] = scores

results_csv = os.path.join(results_dir, "OneClassSVM_Anomaly_Results.csv")
summary_txt = os.path.join(results_dir, "OneClassSVM_Anomaly_Summary.txt")
plot_path = os.path.join(results_dir, "OneClassSVM_Anomaly_Plot.png")

df_results.to_csv(results_csv, index=False)
with open(summary_txt, "w") as f:
    f.write(summary_text)

# =========================
# 7️⃣ Visualization
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(df['SysBP'], df['Pulse'], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.xlabel("Systolic Blood Pressure (SysBP)")
plt.ylabel("Pulse")
plt.title("One-Class SVM Anomaly Detection on ICU Data")
plt.grid(True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 8️⃣ Done
# =========================
print(f"✅ One-Class SVM anomaly detection completed successfully.")
print(f"   - Results: {results_csv}")
print(f"   - Summary: {summary_txt}")
print(f"   - Plot: {plot_path}")
