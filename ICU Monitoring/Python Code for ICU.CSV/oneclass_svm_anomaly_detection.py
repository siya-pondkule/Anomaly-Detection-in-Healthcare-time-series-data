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

multi_summary_csv = os.path.join(results_dir, "OneClassSVM_MultiThreshold_Summary.csv")
best_summary_csv = os.path.join(results_dir, "OneClassSVM_BestThreshold.csv")
results_csv = os.path.join(results_dir, "OneClassSVM_Anomaly_Results.csv")
plot_path = os.path.join(results_dir, "OneClassSVM_Anomaly_Plot.png")

# =========================
# 2️⃣ Load and preprocess data
# =========================
df = pd.read_csv(input_path)
features = ['Age', 'SysBP', 'Pulse']
df = df[features].dropna().reset_index(drop=True)

# Physiological thresholds
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

# Ground truth
gt = np.zeros(len(df))
gt[df["SysBP"] > THRESHOLDS["SBP_high"]] = 1
gt[df["SysBP"] < THRESHOLDS["SBP_low"]] = 1
gt[df["Pulse"] > THRESHOLDS["HR_tachy"]] = 1
gt[df["Pulse"] < THRESHOLDS["HR_brady"]] = 1

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================
# 3️⃣ Train One-Class SVM
# =========================
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
pred_raw = svm.fit_predict(X_scaled)

y_pred_svm = np.where(pred_raw == -1, 1, 0)  # anomaly =1

# Scores
scores = -svm.score_samples(X_scaled)  # higher = more anomalous

# =========================
# 4️⃣ MULTI-THRESHOLD EVALUATION (90 / 95 / 98)
# =========================
thresholds = [90, 95, 98]
rows = []
best_f1 = -1
best_row = None
best_threshold_value = None
best_pred = None

for pct in thresholds:
    thr = np.percentile(scores, pct)
    y_pred = (scores >= thr).astype(int)

    try:
        auc = roc_auc_score(gt, scores)
    except:
        auc = np.nan

    precision = precision_score(gt, y_pred, zero_division=0)
    recall = recall_score(gt, y_pred, zero_division=0)
    f1 = f1_score(gt, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(gt, y_pred).ravel()
    far = fp / (fp + tn) if (fp + tn) else 0

    row = {
        "Threshold(%)": pct,
        "Threshold_Value": thr,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Detected_Anomalies": int(y_pred.sum())
    }
    rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row
        best_threshold_value = thr
        best_pred = y_pred.copy()

# Save multi-threshold summary
pd.DataFrame(rows).to_csv(multi_summary_csv, index=False)

# Save best threshold summary
pd.DataFrame([best_row]).to_csv(best_summary_csv, index=False)

print("\n=== Multi-threshold Evaluation Complete ===")
print(pd.DataFrame(rows))

print("\n=== BEST THRESHOLD (based on F1) ===")
print(best_row)

# =========================
# 5️⃣ Save detailed results
# =========================
df_results = df.copy()
df_results["Score"] = scores
df_results["Predicted_Anomaly"] = best_pred
df_results["GroundTruth"] = gt
df_results.to_csv(results_csv, index=False)

# =========================
# 6️⃣ Plot results with BEST threshold
# =========================
plt.figure(figsize=(10, 6))
plt.scatter(df["SysBP"], df["Pulse"], c=best_pred, cmap="coolwarm", edgecolors="k")
plt.title(f"One-Class SVM - Best Threshold = {best_row['Threshold(%)']}%")
plt.xlabel("SysBP")
plt.ylabel("Pulse")
plt.grid(True)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

# =========================
# 7️⃣ Done
# =========================
print("\nBEST threshold applied on plot + results saved!")
print(f"Multi-threshold summary → {multi_summary_csv}")
print(f"Best-threshold summary → {best_summary_csv}")
print(f"Full results CSV → {results_csv}")
print(f"Plot → {plot_path}")
