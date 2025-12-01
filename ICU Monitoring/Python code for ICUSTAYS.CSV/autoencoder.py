import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\Autoencoder-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

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
# MAIN PROCESSING
# ======================
print(f"\n=== Processing {DATA_PATH.name} ===")
df = pd.read_csv(DATA_PATH)

# Drop timestamp/id/unit columns
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["time", "date", "id", "unit"])]
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
numeric_df = numeric_df.dropna().reset_index(drop=True)

if numeric_df.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv.")

print(f"Using columns for anomaly detection: {list(numeric_df.columns)}")

# Pseudo ground truth (all zeros because no vitals)
gt_array = np.zeros(len(numeric_df))

# ======================
# Scale data
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
input_dim = X_scaled.shape[1]

# ======================
# Train Autoencoder
# ======================
autoencoder = build_autoencoder(input_dim)
history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
print(f"Training complete | Final Loss={history.history['loss'][-1]:.4f}")

# Reconstruction
recon = autoencoder.predict(X_scaled)
mse_scores = np.mean((X_scaled - recon) ** 2, axis=1)

# =======================================================
#  MULTI-THRESHOLD EVALUATION (90, 95, 98)
# =======================================================
thresholds = [90, 95, 98]
eval_rows = []
best_f1 = -1
best_row = None

for pct in thresholds:
    thr = np.percentile(mse_scores, pct)
    y_pred = (mse_scores >= thr).astype(int)

    # With no GT anomalies → metrics comparisons still apply
    try:
        auc = roc_auc_score(gt_array, mse_scores)
    except:
        auc = np.nan

    precision = precision_score(gt_array, y_pred, zero_division=0)
    recall = recall_score(gt_array, y_pred, zero_division=0)
    f1 = f1_score(gt_array, y_pred, zero_division=0)

    cm = confusion_matrix(gt_array, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    far = fp / (fp + tn) if (fp + tn) else 0

    row = {
        "Threshold_%": pct,
        "Threshold_Value": thr,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Detected_Anomalies": int(y_pred.sum())
    }
    eval_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save multi-threshold summary
multi_df = pd.DataFrame(eval_rows)
multi_df.to_csv(SUMMARY_DIR / "ICUSTAYS_Autoencoder_MultiThresholdSummary.csv", index=False)

# Save best-threshold summary
pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "ICUSTAYS_Autoencoder_BestThreshold.csv", index=False)

print("\n=== Multi-threshold evaluation (90/95/98) ===")
print(multi_df)
print("\n=== BEST THRESHOLD (by F1) ===")
print(best_row)

# =======================================================
# Apply BEST threshold
# =======================================================
best_thr = best_row["Threshold_Value"]
final_pred = (mse_scores >= best_thr).astype(int)
anomaly_indices = np.where(final_pred == 1)[0]

# Save anomaly summary (your original requirement)
summary_df = pd.DataFrame({
    "Index": anomaly_indices,
    "Reconstruction_Error": mse_scores[anomaly_indices]
})
summary_df.to_csv(SUMMARY_DIR / "Autoencoder_ICUSTAYS_anomaly_summary.csv", index=False)

print(f"\nFinal anomaly summary saved → {SUMMARY_DIR / 'Autoencoder_ICUSTAYS_anomaly_summary.csv'}")

# =======================================================
# Plot
# =======================================================
plt.figure(figsize=(12,6))
for col in numeric_df.columns:
    plt.plot(numeric_df[col], label=f"{col}", alpha=0.55)

for idx in anomaly_indices:
    plt.axvline(idx, color='red', linestyle='--', alpha=0.3)

plt.title("ICUSTAYS Autoencoder Anomaly Detection")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_AnomalyPlot.png")
plt.close()

print(f"\nResults saved to: {OUTPUT_DIR}")
