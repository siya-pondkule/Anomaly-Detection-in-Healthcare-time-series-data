import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

print("\n=== Training TCN Autoencoder anomaly detection model on ICU.csv ===")

# =========================
# Paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\TCN-ModelResult"
os.makedirs(results_dir, exist_ok=True)

summary_multi = os.path.join(results_dir, "TCN_MultiThreshold_Summary.csv")
summary_best = os.path.join(results_dir, "TCN_BestThreshold.csv")
detailed_csv = os.path.join(results_dir, "TCN_Autoencoder_Detailed_Results.csv")
hist_path = os.path.join(results_dir, "TCN_ReconstructionError_Hist.png")

# =========================
# Physiological thresholds
# =========================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR"],
    "SBP": ["SysBP", "SBP", "SystolicBP"]
}

def label_anomalies(series, col_name, gt_array):
    series_num = pd.to_numeric(series, errors="coerce").reset_index(drop=True)
    if series_num.dropna().empty:
        return

    vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)

    if vital == "HR":
        gt_array[series_num > THRESHOLDS["HR_tachy"]] = 1
        gt_array[series_num < THRESHOLDS["HR_brady"]] = 1

    elif vital == "SBP":
        gt_array[series_num > THRESHOLDS["SBP_high"]] = 1
        gt_array[series_num < THRESHOLDS["SBP_low"]] = 1

# =========================
# TCN Autoencoder
# =========================
def build_tcn_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, 3, padding="same", activation="relu", dilation_rate=1)(inputs)
    x = Conv1D(16, 3, padding="same", activation="relu", dilation_rate=2)(x)
    encoded = Conv1D(8, 3, padding="same", activation="relu", dilation_rate=4)(x)

    x = Conv1D(16, 3, padding="same", activation="relu", dilation_rate=2)(encoded)
    x = Conv1D(32, 3, padding="same", activation="relu", dilation_rate=1)(x)
    decoded = Conv1D(input_shape[-1], 3, padding="same")(x)

    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

# =========================
# Load Data
# =========================
df = pd.read_csv(input_path)
features = ['Age', 'SysBP', 'Pulse']
df = df[features].dropna().reset_index(drop=True)

# Ground Truth
gt = np.zeros(len(df))
for col in df.columns:
    label_anomalies(df[col], col, gt)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Reshape for Conv1D
X_in = np.expand_dims(X_scaled, axis=1)

# =========================
# Train model
# =========================
print("\n=== Training TCN Autoencoder ===")
model = build_tcn_autoencoder(X_in.shape[1:])
model.fit(
    X_in, X_in,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# =========================
# Reconstruction error
# =========================
recon = model.predict(X_in)
mse = np.mean((X_in - recon) ** 2, axis=(1, 2))

# =========================
# MULTI-THRESHOLD EVALUATION (90 / 95 / 98 %)
# =========================
threshold_list = [90, 95, 98]
results = []

best_f1 = -1
best_info = None
best_pred = None

for pct in threshold_list:
    thr = np.percentile(mse, pct)
    pred = (mse >= thr).astype(int)

    try:
        auc = roc_auc_score(gt, mse)
    except:
        auc = np.nan

    precision = precision_score(gt, pred, zero_division=0)
    recall = recall_score(gt, pred, zero_division=0)
    f1 = f1_score(gt, pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
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
        "Detected_Anomalies": int(pred.sum())
    }
    results.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_info = row
        best_pred = pred.copy()

# =========================
# Save multi-threshold summary
# =========================
pd.DataFrame(results).to_csv(summary_multi, index=False)

# Save best threshold summary
pd.DataFrame([best_info]).to_csv(summary_best, index=False)

# =========================
# Save detailed per sample
# =========================
detailed_df = df.copy()
detailed_df["Reconstruction_Error"] = mse
detailed_df["GT_Anomaly"] = gt
detailed_df["Pred_Anomaly"] = best_pred

detailed_df.to_csv(detailed_csv, index=False)

# =========================
# Plot
# =========================
plt.figure(figsize=(8, 4))
plt.hist(mse, bins=80, alpha=0.7)
plt.axvline(best_info["Threshold_Value"], color="red", linestyle="--",
            label=f"Best {best_info['Threshold(%)']}% Threshold")
plt.title("TCN Reconstruction Error Distribution")
plt.xlabel("MSE")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(hist_path, dpi=200)
plt.close()

print("\n===== MULTI-THRESHOLD RESULTS =====")
print(pd.DataFrame(results))

print("\n===== BEST THRESHOLD ==== ")
print(best_info)

print("\nFiles Saved:")
print(f" - Multi-threshold summary → {summary_multi}")
print(f" - Best threshold summary → {summary_best}")
print(f" - Detailed results → {detailed_csv}")
print(f" - Histogram → {hist_path}")

print("\nTCN Autoencoder multi-threshold evaluation complete!")
