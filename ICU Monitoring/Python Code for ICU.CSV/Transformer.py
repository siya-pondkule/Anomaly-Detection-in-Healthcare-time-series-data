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

multi_summary_path = os.path.join(results_dir, "Transformer_MultiThreshold_Summary.csv")
best_summary_path = os.path.join(results_dir, "Transformer_BestThreshold.csv")
detailed_path = os.path.join(results_dir, "Transformer_Autoencoder_Detailed_Results.csv")

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

def label_anomalies(series, col_name, gt_array, offset=0):
    s = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if s.empty:
        return
    vital = next((v for v,cols in VITAL_MAPPING.items() if col_name in cols),None)

    if vital == "HR":
        gt_array[s > THRESHOLDS["HR_tachy"]] = 1
        gt_array[s < THRESHOLDS["HR_brady"]] = 1
    elif vital == "SBP":
        gt_array[s > THRESHOLDS["SBP_high"]] = 1
        gt_array[s < THRESHOLDS["SBP_low"]] = 1

# =========================
# Transformer Block
# =========================
def transformer_block(inputs, heads=2, ff_dim=64, drop=0.1):
    attn = MultiHeadAttention(num_heads=heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn = Dropout(drop)(attn)
    out1 = LayerNormalization()(inputs + attn)

    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn = Dropout(drop)(ffn)
    out2 = LayerNormalization()(out1 + ffn)
    return out2

def build_transformer_autoencoder(input_shape):
    input_layer = Input(shape=input_shape)
    x = transformer_block(input_layer)
    x = transformer_block(x)
    encoded = Dense(16, activation="relu")(x)
    x = transformer_block(encoded)
    out = Dense(input_shape[-1])(x)
    model = Model(input_layer, out)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

# =========================
# Load Data
# =========================
df = pd.read_csv(input_path)
features = ['Age','SysBP','Pulse']
df = df[features].dropna().reset_index()

gt = np.zeros(len(df))
for col in df.columns:
    label_anomalies(df[col], col, gt)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# reshape into sequence length 1
X_seq = np.expand_dims(X_scaled, axis=1)

# =========================
# Training
# =========================
print("\nTraining Transformer Autoencoder …")
model = build_transformer_autoencoder(X_seq.shape[1:])
model.fit(X_seq, X_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# =========================
# Reconstruction Error
# =========================
recon = model.predict(X_seq)
mse = np.mean((X_seq - recon)**2, axis=(1,2))

# =========================
# MULTI-THRESHOLD EVALUATION (90/95/98)
# =========================
thresholds = [90, 95, 98]
results = []

best_f1 = -1
best_info = None
best_pred = None

for p in thresholds:
    thr = np.percentile(mse, p)
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
        "Threshold_%": p,
        "Threshold_Value": thr,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Detected_Anomalies": int(pred.sum())
    }
    results.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_info = row
        best_pred = pred.copy()

# =========================
# Save CSVs
# =========================
pd.DataFrame(results).to_csv(multi_summary_path, index=False)
pd.DataFrame([best_info]).to_csv(best_summary_path, index=False)

# detailed results
detailed_df = df.copy()
detailed_df["Reconstruction_Error"] = mse
detailed_df["GT"] = gt
detailed_df["Pred"] = best_pred
detailed_df.to_csv(detailed_path, index=False)

# =========================
# Print Results
# =========================
print("\n=== MULTI-THRESHOLD RESULTS (90/95/98) ===")
print(pd.DataFrame(results))

print("\n=== BEST THRESHOLD ===")
print(best_info)

print("\nSaved:")
print(multi_summary_path)
print(best_summary_path)
print(detailed_path)
