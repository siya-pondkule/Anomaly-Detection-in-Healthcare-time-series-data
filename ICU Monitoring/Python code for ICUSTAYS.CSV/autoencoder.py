import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
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
# Evaluation Function
# ======================
def evaluate_anomaly_detection(y_true, mse_scores):
    try:
        auc = roc_auc_score(y_true, mse_scores)
    except ValueError:
        auc = np.nan

    best_f1, best_threshold = 0, np.percentile(mse_scores, 90)
    for q in np.linspace(90, 99.9, 100):
        threshold = np.percentile(mse_scores, q)
        y_pred = (mse_scores >= threshold).astype(int)
        if np.sum(y_pred) > 0 and np.sum(y_true) > 0:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold

    y_pred_best = (mse_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "AUC": auc,
        "F1": best_f1,
        "Precision": precision_score(y_true, y_pred_best, zero_division=0),
        "Recall": recall_score(y_true, y_pred_best, zero_division=0),
        "FAR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "Threshold": best_threshold
    }


# ======================
# Main Processing
# ======================
print(f"\n=== Processing {DATA_PATH.name} ===")
df = pd.read_csv(DATA_PATH)

# Drop non-numeric and identifier/timestamp columns
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["time", "date", "id", "unit"])]
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
numeric_df = numeric_df.dropna().reset_index(drop=True)

if numeric_df.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for anomaly detection.")

print(f"Using columns for anomaly detection: {list(numeric_df.columns)}")

# Since ICUSTAYS.csv doesn’t have explicit vitals, we assume anomalies are outliers in LOS/durations etc.
# Create pseudo ground truth (for evaluation purposes)
gt_array = np.zeros(len(numeric_df))

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)
input_dim = X_scaled.shape[1]

# Train Autoencoder
autoencoder = build_autoencoder(input_dim)
history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
print(f"Training complete | Final Loss: {history.history['loss'][-1]:.4f}")

# Reconstruction and anomaly scoring
reconstructions = autoencoder.predict(X_scaled)
mse_per_row = np.mean(np.square(X_scaled - reconstructions), axis=1)

# Identify anomalies based on high reconstruction error
threshold = np.percentile(mse_per_row, 95)
anomaly_indices = np.where(mse_per_row > threshold)[0]

# Evaluate (no ground truth, so we skip F1-based tuning)
y_pred = np.zeros(len(mse_per_row))
y_pred[anomaly_indices] = 1

print(f"\nDetected {len(anomaly_indices)} anomalies out of {len(mse_per_row)} records.")

# Reconstruction performance
mse_reco = mean_squared_error(X_scaled, reconstructions)
r2 = r2_score(X_scaled, reconstructions)
accuracy = 100 * (1 - mse_reco)
print(f"Reconstruction: MSE={mse_reco:.4f} | R²={r2:.4f} | Accuracy≈{accuracy:.2f}%")

# Save corrected data
df_corrected = pd.DataFrame(scaler.inverse_transform(reconstructions), columns=numeric_df.columns)
df_corrected.to_csv(OUTPUT_DIR / "ICUSTAYS_corrected.csv", index=False)

# ======================
# Summary of Anomalies
# ======================
summary_df = pd.DataFrame({
    "Index": anomaly_indices,
    "Reconstruction_Error": mse_per_row[anomaly_indices]
})
summary_df.to_csv(SUMMARY_DIR / "Autoencoder_ICUSTAYS_anomaly_summary.csv", index=False)
print(f"\nAnomaly summary saved → {SUMMARY_DIR / 'Autoencoder_ICUSTAYS_anomaly_summary.csv'}")

# ======================
# Plot anomalies
# ======================
plt.figure(figsize=(12,6))
for col in numeric_df.columns:
    plt.plot(numeric_df[col], label=f"{col}", alpha=0.6)
for idx in anomaly_indices:
    plt.axvline(idx, color='red', linestyle='--', alpha=0.3)
plt.title("ICUSTAYS Anomaly Detection (Autoencoder)")
plt.xlabel("Record Index")
plt.ylabel("Value")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_anomaly_plot.png")
plt.close()

print(f"\nResults saved to → {OUTPUT_DIR}")
