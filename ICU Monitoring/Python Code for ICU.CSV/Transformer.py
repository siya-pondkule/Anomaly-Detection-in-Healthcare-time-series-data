import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

# =========================
# Config / Paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\Transformer-ModelResult"
os.makedirs(results_dir, exist_ok=True)

# =========================
# Physiological thresholds (OPTION A)
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
    """Label physiological anomalies in `series` and update gt_array in-place.
       Returns list of anomaly dicts (for optional plotting)."""
    s = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if s.empty:
        return []
    vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
    anomalous_indices = []
    if vital == "HR":
        anomalous_indices += list(s[s > THRESHOLDS["HR_tachy"]].index)
        anomalous_indices += list(s[s < THRESHOLDS["HR_brady"]].index)
    elif vital == "SBP":
        anomalous_indices += list(s[s > THRESHOLDS["SBP_high"]].index)
        anomalous_indices += list(s[s < THRESHOLDS["SBP_low"]].index)

    for i in anomalous_indices:
        if (i + index_offset) < len(gt_array):
            gt_array[i + index_offset] = 1

    return [{"vital": col_name, "index": i + index_offset} for i in anomalous_indices]

# =========================
# Evaluation helper
# =========================
def evaluate_with_threshold_search(y_true, scores, q_min=90.0, q_max=99.9, q_steps=100):
    """Find best threshold by scanning percentiles between q_min and q_max.
       Returns dict with AUC, best F1, Precision, Recall, FAR and best threshold."""
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = np.nan

    best_f1 = -1.0
    best_thresh = np.percentile(scores, q_min)
    for q in np.linspace(q_min, q_max, q_steps):
        thresh = np.percentile(scores, q)
        y_pred = (scores >= thresh).astype(int)
        if y_pred.sum() == 0:
            continue
        # require at least one positive in y_true to compute meaningful F1
        if y_true.sum() == 0:
            f1 = 0.0
        else:
            f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # final metrics at best_thresh
    y_pred_best = (scores >= best_thresh).astype(int)
    try:
        cm = confusion_matrix(y_true, y_pred_best)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    except Exception:
        tn = fp = fn = tp = 0

    precision = precision_score(y_true, y_pred_best, zero_division=0)
    recall = recall_score(y_true, y_pred_best, zero_division=0)
    f1 = f1_score(y_true, y_pred_best, zero_division=0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "AUC": auc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "FAR": far,
        "Best_Threshold": float(best_thresh)
    }

# =========================
# Transformer Autoencoder (model)
# =========================
def transformer_block(inputs, num_heads, ff_dim, dropout=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn = Dropout(dropout)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2

def build_transformer_autoencoder(input_shape, num_heads=2, ff_dim=64):
    inputs = Input(shape=input_shape)
    x = transformer_block(inputs, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)
    encoded = Dense(16, activation='relu')(x)
    x = transformer_block(encoded, num_heads=num_heads, ff_dim=ff_dim)
    decoded = Dense(input_shape[-1], activation=None)(x)
    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

# =========================
# MAIN
# =========================
print("\n=== Training Transformer Autoencoder on ICU.csv (using physiological labels) ===")
df = pd.read_csv(input_path)

# keep same features as your other models
features = ['Age', 'SysBP', 'Pulse']
df = df[features].dropna().reset_index(drop=True)

# create ground-truth labels using physiological thresholds (OPTION A)
gt_array = np.zeros(len(df), dtype=int)
anomaly_list = []
for col in df.columns:
    anomaly_list.extend(label_anomalies(df[col], col, gt_array, 0))

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# reshape to (samples, timesteps, features) — keep 1 timestep per record (as earlier)
X_seq = np.expand_dims(X_scaled, axis=1)  # shape (n_samples, 1, n_features)

# build & train model
model = build_transformer_autoencoder(X_seq.shape[1:])
model.summary()

EPOCHS = 50
BATCH_SIZE = 32

history = model.fit(
    X_seq, X_seq,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# reconstructions and per-sample MSE
reconstructions = model.predict(X_seq)
mse = np.mean(np.power(X_seq - reconstructions, 2), axis=(1, 2))  # one MSE per sample

# Evaluate: find best threshold by percentile search and compute metrics vs gt_array
metrics = evaluate_with_threshold_search(gt_array, mse, q_min=90.0, q_max=99.9, q_steps=200)

# produce anomaly indices using best threshold
best_thresh = metrics["Best_Threshold"]
y_pred = (mse >= best_thresh).astype(int)
anomaly_indices = np.where(y_pred == 1)[0]

# print results
print("\n=== Transformer Autoencoder Evaluation (physiological GT) ===")
print(f"AUC = {metrics['AUC']:.4f}")
print(f"F1  = {metrics['F1']:.4f}")
print(f"Precision = {metrics['Precision']:.4f}")
print(f"Recall = {metrics['Recall']:.4f}")
print(f"FAR = {metrics['FAR']:.4f}")
print(f"Best Threshold (MSE) = {metrics['Best_Threshold']:.6e}")
print(f"Detected anomaly count = {len(anomaly_indices)}")

# =========================
# Save results & summary CSV (with requested fields)
# =========================
results_df = df.copy()
results_df["Reconstruction_Error"] = mse
results_df["Anomaly_Pred"] = y_pred
results_df["Anomaly_GT"] = gt_array

results_csv = os.path.join(results_dir, "Transformer_Autoencoder_Results.csv")
results_df.to_csv(results_csv, index=False)

summary_df = pd.DataFrame([{
    "Dataset": os.path.basename(input_path),
    "Model": "Transformer_Autoencoder",
    "AUC": metrics["AUC"],
    "F1": metrics["F1"],
    "Precision": metrics["Precision"],
    "Recall": metrics["Recall"],
    "FAR": metrics["FAR"],
    "Best_Threshold": metrics["Best_Threshold"],
    "Num_Anomalies_Detected": int(len(anomaly_indices))
}])

summary_csv = os.path.join(results_dir, "Transformer_Autoencoder_Summary.csv")
summary_df.to_csv(summary_csv, index=False)

# save anomaly indices list for convenience
anomaly_idx_file = os.path.join(results_dir, "Transformer_Autoencoder_Anomaly_Indices.csv")
pd.DataFrame({"Anomaly_Index": anomaly_indices}).to_csv(anomaly_idx_file, index=False)

# =========================
# Plot (scatter of SysBP vs Pulse marking anomalies)
# =========================
plt.figure(figsize=(8,6))
plt.scatter(df["SysBP"], df["Pulse"], c=results_df["Anomaly_Pred"], cmap="coolwarm", edgecolors='k')
plt.xlabel("Systolic Blood Pressure (SysBP)")
plt.ylabel("Pulse")
plt.title("Transformer Autoencoder: Detected Anomalies (red)")
plt.grid(True)
plot_path = os.path.join(results_dir, "Transformer_Autoencoder_AnomalyPlot.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Saved results to: {results_csv}")
print(f"✅ Saved summary to: {summary_csv}")
print(f"✅ Saved anomaly indices to: {anomaly_idx_file}")
print(f"✅ Plot saved to: {plot_path}")
