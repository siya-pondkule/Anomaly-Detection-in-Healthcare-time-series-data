import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------
# Paths
# -----------------------
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\Transformer-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for Transformer Autoencoder ===")

# -----------------------
# Encoding & delimiter detection
# -----------------------
try:
    with open(DATA_PATH, "rb") as f:
        sample = f.read(4096)
    encoding = "utf-8"
    sample.decode(encoding)
except Exception:
    encoding = "latin1"
print(f"Detected encoding: {encoding}")

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    lines = [next(f) for _ in range(10)]
sample_text = "\n".join(lines)
delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: sample_text.count(d))
print(f"Detected delimiter: '{best_delim}'")

# -----------------------
# Load CSV
# -----------------------
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"Loaded data shape: {df.shape}")

# -----------------------
# Preprocessing
# -----------------------
for c in df.columns:
    if "time" in c.lower() or "date" in c.lower():
        df[c] = pd.to_datetime(df[c], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"].fillna(0.0, inplace=True)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.insert(0, "time_diff_min")

df_num = df[numeric_cols].copy()
df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
df_num.fillna(df_num.mean(), inplace=True)

df_num = df_num.loc[:, df_num.apply(pd.Series.nunique) > 1]

if df_num.empty:
    raise ValueError("No usable numeric columns found.")

# -----------------------
# Scaling
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# -----------------------
# Create sliding sequences
# -----------------------
WINDOW = 20
X_seq = []
for i in range(len(X_scaled) - WINDOW + 1):
    X_seq.append(X_scaled[i:i+WINDOW])
X_seq = np.array(X_seq)

timesteps, n_features = X_seq.shape[1], X_seq.shape[2]

# -----------------------
# Transformer Block
# -----------------------
def transformer_encoder_block(x, num_heads=4, head_dim=16, ff_dim=64, dropout_rate=0.1):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=head_dim)(x, x)
    attn = Dropout(dropout_rate)(attn)
    x = LayerNormalization(epsilon=1e-6)(x + attn)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)
    ff = Dropout(dropout_rate)(ff)
    x = LayerNormalization(epsilon=1e-6)(x + ff)
    return x

# -----------------------
# Transformer Autoencoder
# -----------------------
inputs = Input(shape=(timesteps, n_features))

x = transformer_encoder_block(inputs, 4, 16, 128, 0.1)
x = transformer_encoder_block(x, 4, 16, 128, 0.1)

encoded = GlobalAveragePooling1D()(x)

latent = Dense(64, activation="relu")(encoded)

proj = Dense(timesteps * n_features)(latent)
decoded = Reshape((timesteps, n_features))(proj)

decoded = transformer_encoder_block(decoded, 2, 16, 64, 0.05)
outputs = Dense(n_features, activation="linear")(decoded)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-3), loss="mse")

# -----------------------
# Train
# -----------------------
es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

history = model.fit(
    X_seq, X_seq,
    epochs=50, batch_size=128,
    validation_split=0.1,
    shuffle=True,
    callbacks=[es],
    verbose=1
)

# -----------------------
# Reconstruction error
# -----------------------
X_pred = model.predict(X_seq)
mse_seq = np.mean(np.square(X_seq - X_pred), axis=(1,2))

# ==========================================================
# 🔥 MULTI-THRESHOLD EVALUATION (90–95–98)
# ==========================================================
thresholds = [90, 95, 98]
eval_rows = []
best_f1 = -1
best_row = None

# Pseudo-GT based on 95%
gt = (mse_seq >= np.percentile(mse_seq, 95)).astype(int)

# AUC
try:
    auc_value = roc_auc_score(gt, mse_seq)
except:
    auc_value = np.nan

for q in thresholds:
    thr = np.percentile(mse_seq, q)
    pred = (mse_seq >= thr).astype(int)

    precision = precision_score(gt, pred, zero_division=0)
    recall = recall_score(gt, pred, zero_division=0)
    f1 = f1_score(gt, pred, zero_division=0)

    cm = confusion_matrix(gt, pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = (0,0,0,0)

    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    row = {
        "Threshold_percentile": q,
        "Threshold_value": thr,
        "AUC": auc_value,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Detected_Anomalies": pred.sum()
    }

    eval_rows.append(row)

    if f1 > best_f1:
        best_f1 = f1
        best_row = row.copy()

# Save Multi-threshold summary
multi_df = pd.DataFrame(eval_rows)
multi_df.to_csv(SUMMARY_DIR / "Transformer_LABEVENTS_MultiThresholdSummary.csv", index=False)

# Save BEST threshold
pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "Transformer_LABEVENTS_BestThreshold.csv", index=False)

print("\n=== Multi-threshold Summary ===")
print(multi_df)

print("\n=== BEST Threshold ===")
print(best_row)

# -----------------------
# Save normal outputs (unchanged pipeline)
# -----------------------
threshold_95 = np.percentile(mse_seq, 95)
anomaly_mask = mse_seq > threshold_95

results_df = pd.DataFrame({
    "sequence_index": np.arange(len(mse_seq)),
    "reconstruction_error": mse_seq,
    "is_anomaly": anomaly_mask.astype(int)
})
results_df.to_csv(OUTPUT_DIR / "LABEVENTS_Transformer_Results.csv", index=False)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(12,6))
plt.plot(mse_seq, label="Reconstruction Error", alpha=0.7)
plt.axhline(best_row["Threshold_value"], color='orange', linestyle="--",
            label=f"BEST Threshold ({best_row['Threshold_percentile']}%)")
plt.scatter(np.where(anomaly_mask)[0], mse_seq[anomaly_mask], color='red', label="Anomaly")
plt.title("Transformer Autoencoder - LABEVENTS")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_Transformer_AnomalyPlot.png")
plt.close()

print("\n🎯 Transformer anomaly detection + multi-threshold evaluation completed!")
print("✔ Multi-threshold summary saved")
print("✔ Best threshold summary saved")
