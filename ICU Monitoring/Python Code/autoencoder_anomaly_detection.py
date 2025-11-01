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
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\Autoencoder-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary"  # parent directory of OUTPUT_DIR + /Summary
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Thresholds for vital signs (only those present)
# ======================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

# Vital name mapping for your dataset
VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR"],
    "SBP": ["SysBP", "SBP", "SystolicBP"]
}

# ======================
# Detect and LABEL anomalies (Ground Truth)
# ======================
def label_anomalies(series, col_name, gt_array, index_offset):
    """Detects physiological anomalies and updates the Ground Truth array."""
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
df = df[["SysBP", "Pulse"]]  # Keep only relevant columns
df = df.dropna().reset_index(drop=True)

# Ground Truth labeling
gt_array = np.zeros(len(df))
anomalies = []
for col in df.columns:
    anomalies.extend(label_anomalies(df[col], col, gt_array, 0))

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
input_dim = X_scaled.shape[1]

# Train Autoencoder
autoencoder = build_autoencoder(input_dim)
history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
print(f"Training complete | Final Loss: {history.history['loss'][-1]:.4f}")

# Reconstruction and evaluation
reconstructions = autoencoder.predict(X_scaled)
mse_per_row = np.mean(np.square(X_scaled - reconstructions), axis=1)
eval_results = evaluate_anomaly_detection(gt_array, mse_per_row)

print(f"\nResults for ICU.csv:")
print(f"AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | "
      f"Precision={eval_results['Precision']:.4f} | Recall={eval_results['Recall']:.4f} | "
      f"FAR={eval_results['FAR']:.4f}")

# Reconstruction performance
mse_reco = mean_squared_error(X_scaled, reconstructions)
r2 = r2_score(X_scaled, reconstructions)
accuracy = 100 * (1 - mse_reco)
print(f"Reconstruction: MSE={mse_reco:.4f} | R²={r2:.4f} | Accuracy≈{accuracy:.2f}%")

# Save corrected (reconstructed) data
df_corrected = pd.DataFrame(scaler.inverse_transform(reconstructions), columns=df.columns)
df_corrected.to_csv(OUTPUT_DIR / "ICU_corrected.csv", index=False)

# ======================
# Summary: Count and Save Anomalies
# ======================
summary_data = []
for vital in df.columns:
    indices = [a["index"] for a in anomalies if a["vital"] == vital]
    summary_data.append({
        "Vital Sign": vital,
        "Total Anomalies": len(indices),
        "Anomaly Record Indices": ", ".join(map(str, indices)) if indices else "None"
    })

summary_df = pd.DataFrame(summary_data)
summary_file = SUMMARY_DIR / "Autoencoder_anomaly_summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"\nAnomaly summary saved → {summary_file}")

# ======================
# Plot anomalies
# ======================
plt.figure(figsize=(12,6))
plt.plot(df["SysBP"], label="SysBP (Original)", alpha=0.7)
plt.plot(df_corrected["SysBP"], label="SysBP (Reconstructed)", alpha=0.9)
plt.plot(df["Pulse"], label="Pulse (Original)", alpha=0.7)
plt.plot(df_corrected["Pulse"], label="Pulse (Reconstructed)", alpha=0.9)
for anom in anomalies:
    plt.scatter(anom["index"], df.loc[anom["index"], anom["vital"]],
                color="red", marker="x", s=50)
plt.title("ICU Anomaly Detection (SysBP & Pulse)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICU_anomaly_plot.png")
plt.close()

print(f"\nResults saved to → {OUTPUT_DIR}")
