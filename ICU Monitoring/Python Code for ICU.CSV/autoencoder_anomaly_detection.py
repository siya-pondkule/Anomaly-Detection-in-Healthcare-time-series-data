import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\Autoencoder-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Vital thresholds
# ======================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

# Vital name mapping
VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR"],
    "SBP": ["SysBP", "SBP", "SystolicBP"]
}

# ======================
# Ground Truth Labeling
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

    return anomalous_indices

# ======================
# Autoencoder Model
# ======================
def build_autoencoder(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu', name='encoded'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ======================
# Metric Calculation for Thresholds
# ======================
def compute_metrics(y_true, y_pred, mse_scores):
    try:
        auc = roc_auc_score(y_true, mse_scores)
    except:
        auc = np.nan

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    return precision, recall, f1, auc, far, tn, fp, fn, tp

# ======================
# Load Data
# ======================
print(f"\n=== Processing {DATA_PATH.name} ===")
df = pd.read_csv(DATA_PATH)
df = df[["SysBP", "Pulse"]].dropna().reset_index(drop=True)

# ======================
# Ground Truth
# ======================
gt_array = np.zeros(len(df))
for col in df.columns:
    label_anomalies(df[col], col, gt_array, 0)

# ======================
# Scale Data
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
input_dim = X_scaled.shape[1]

# ======================
# Train Autoencoder
# ======================
autoencoder = build_autoencoder(input_dim)
history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50, batch_size=32,
    validation_split=0.1, verbose=0
)

# ======================
# Reconstruction Errors
# ======================
recon = autoencoder.predict(X_scaled)
mse_scores = np.mean((X_scaled - recon)**2, axis=1)

# ======================
# Evaluate at fixed thresholds 90/95/98
# ======================
thresholds = [90, 95, 98]
threshold_results = []
best_f1 = -1
best_row = None

for pct in thresholds:
    thr = np.percentile(mse_scores, pct)
    y_pred = (mse_scores >= thr).astype(int)
    y_true = gt_array.copy()

    precision, recall, f1, auc, far, tn, fp, fn, tp = compute_metrics(y_true, y_pred, mse_scores)

    row = {
        "Percentile": pct,
        "Threshold_Value": thr,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "GT_Anomalies": int(y_true.sum()),
        "Predicted_Anomalies": int(y_pred.sum()),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }
    threshold_results.append(row)

    # Select best threshold (based on F1)
    if f1 > best_f1:
        best_f1 = f1
        best_row = row

# ======================
# Save Threshold Comparison
# ======================
th_df = pd.DataFrame(threshold_results)
th_df.to_csv(SUMMARY_DIR / "Autoencoder_ICU_Thresholds_90_95_98.csv", index=False)

# ======================
# Save Best Threshold
# ======================
pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "Autoencoder_ICU_BestThreshold.csv", index=False)

print("\n=== BEST THRESHOLD (among 90/95/98) ===")
print(best_row)

# ======================
# Corrected data save
# ======================
df_corrected = pd.DataFrame(scaler.inverse_transform(recon), columns=df.columns)
df_corrected.to_csv(OUTPUT_DIR / "ICU_corrected.csv", index=False)

# ======================
# Plot MSE with thresholds
# ======================
plt.figure(figsize=(12,6))
plt.plot(mse_scores, label="Reconstruction Error")

for pct in thresholds:
    thr = np.percentile(mse_scores, pct)
    plt.axhline(thr, linestyle="--", label=f"{pct}th = {thr:.4f}")

# Mark anomalies from BEST threshold
best_thr = best_row["Threshold_Value"]
best_pred = (mse_scores >= best_thr).astype(int)
plt.scatter(np.where(best_pred==1)[0], mse_scores[best_pred==1], c="red", marker="x", label="Anomalies")

plt.title("Autoencoder ICU Anomaly Detection\n(Threshold Evaluation 90/95/98)")
plt.xlabel("Index")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICU_Autoencoder_Thresholds_Plot.png")
plt.close()

print("\nProcess Completed Successfully!")
