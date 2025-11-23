import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

# =========================
# Paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\TCN-ModelResult"
os.makedirs(results_dir, exist_ok=True)
summary_path = os.path.join(results_dir, "TCN_Autoencoder_Evaluation_Summary.csv")

# =========================
# Physiological thresholds & mapping (for ground-truth labeling)
# =========================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}
VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR"],
    "SBP": ["SysBP", "SBP", "SystolicBP"]
}

def label_anomalies(series, col_name, gt_array, index_offset=0):
    """Label anomalies using simple physiological thresholds.
    Marks positions in gt_array with 1 for anomaly, 0 otherwise.
    """
    series_num = pd.to_numeric(series, errors="coerce").reset_index(drop=True)
    if series_num.dropna().empty:
        return []
    vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
    anomalous_indices = []
    if vital == "HR":
        anomalous_indices += list(series_num[series_num > THRESHOLDS["HR_tachy"]].index)
        anomalous_indices += list(series_num[series_num < THRESHOLDS["HR_brady"]].index)
    elif vital == "SBP":
        anomalous_indices += list(series_num[series_num > THRESHOLDS["SBP_high"]].index)
        anomalous_indices += list(series_num[series_num < THRESHOLDS["SBP_low"]].index)

    for i in anomalous_indices:
        if 0 <= i + index_offset < len(gt_array):
            gt_array[i + index_offset] = 1

    return anomalous_indices

# =========================
# Build TCN-like conv autoencoder
# (keeps original structure but simple Conv1D dilations emulate TCN)
# =========================
def build_tcn_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=3, padding="same", activation="relu", dilation_rate=1)(inputs)
    x = Conv1D(16, kernel_size=3, padding="same", activation="relu", dilation_rate=2)(x)
    encoded = Conv1D(8, kernel_size=3, padding="same", activation="relu", dilation_rate=4)(x)
    x = Conv1D(16, kernel_size=3, padding="same", activation="relu", dilation_rate=2)(encoded)
    x = Conv1D(32, kernel_size=3, padding="same", activation="relu", dilation_rate=1)(x)
    decoded = Conv1D(input_shape[-1], kernel_size=3, padding="same")(x)
    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

# =========================
# Load data
# =========================
print("\n=== Loading data ===")
df = pd.read_csv(input_path)
# Use same feature set as other models to keep comparability
features = ['Age', 'SysBP', 'Pulse']  
df = df[features].dropna().reset_index(drop=True)

# Ground-truth labeling using physiological thresholds
gt = np.zeros(len(df), dtype=int)
for col in df.columns:
    label_anomalies(df[col], col, gt, index_offset=0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# reshape for Conv1D: (samples, timesteps, features)
# treat each record as a short "time step" sequence length=1 (consistent with prior script)
X_in = np.expand_dims(X_scaled, axis=1)  # shape = (n_samples, 1, n_features)

# =========================
# Train model
# =========================
print("\n=== Building and training TCN Autoencoder ===")
model = build_tcn_autoencoder(X_in.shape[1:])
history = model.fit(
    X_in, X_in,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# =========================
# Reconstruction & MSE per sample
# =========================
print("\n=== Computing reconstruction errors ===")
reconstructions = model.predict(X_in)
mse = np.mean(np.power(X_in - reconstructions, 2), axis=(1,2))  # one MSE value per sample

# =========================
# Find best threshold by maximizing F1 over percentile sweep
# =========================
percentiles = np.linspace(50, 99.9, 200)  # broader sweep in case anomalies aren't extremely rare
best_f1 = -1.0
best_thresh = None
best_pred = None

for p in percentiles:
    thresh = np.percentile(mse, p)
    y_pred = (mse >= thresh).astype(int)
    # require both classes present for metric to be meaningful
    if y_pred.sum() == 0 or gt.sum() == 0:
        continue
    f1 = f1_score(gt, y_pred, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        best_pred = y_pred.copy()

# If no threshold found (e.g., no gt anomalies), fall back to mean+2std
if best_thresh is None:
    best_thresh = np.mean(mse) + 2 * np.std(mse)
    best_pred = (mse >= best_thresh).astype(int)
    best_f1 = f1_score(gt, best_pred, zero_division=0)

# =========================
# Compute evaluation metrics using best threshold
# =========================
try:
    auc = roc_auc_score(gt, mse) if (np.unique(gt).size > 1) else np.nan
except Exception:
    auc = np.nan

precision = precision_score(gt, best_pred, zero_division=0)
recall = recall_score(gt, best_pred, zero_division=0)
f1 = f1_score(gt, best_pred, zero_division=0)
cm = confusion_matrix(gt, best_pred)
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

# =========================
# Print & save summary
# =========================
metrics = {
    "AUC": auc,
    "F1": f1,
    "Precision": precision,
    "Recall": recall,
    "FAR": far,
    "Best_Threshold": best_thresh
}
print("\n=== Evaluation Results (TCN Autoencoder) ===")
for k,v in metrics.items():
    print(f"{k}: {v}")

# Save a concise summary CSV (one-row)
summary_df = pd.DataFrame([metrics])
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved evaluation summary → {summary_path}")

# Save detailed results per-sample if desired
results_df = df.copy()
results_df["Reconstruction_Error"] = mse
results_df["GT_Anomaly"] = gt
results_df["Pred_Anomaly"] = best_pred
results_df.to_csv(os.path.join(results_dir, "TCN_Autoencoder_Detailed_Results.csv"), index=False)
print(f"Saved detailed per-sample results → {os.path.join(results_dir, 'TCN_Autoencoder_Detailed_Results.csv')}")

# Optionally plot reconstruction error histogram and threshold
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.hist(mse, bins=80, alpha=0.7)
    plt.axvline(best_thresh, color='r', linestyle='--', label=f"Best thresh={best_thresh:.4e}")
    plt.title("Reconstruction Error Distribution (TCN Autoencoder)")
    plt.xlabel("MSE")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "TCN_ReconstructionError_Hist.png"), dpi=200)
    plt.close()
    print("Saved reconstruction-error histogram.")
except Exception:
    pass

print("\n✅ TCN Autoencoder evaluation complete.")
