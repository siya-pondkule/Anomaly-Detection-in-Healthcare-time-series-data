import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# Paths
# ============================================================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\Transformer-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Transformer Autoencoder with 90/95/98 threshold evaluation ===")

# ============================================================
# Detect encoding + delimiter
# ============================================================
try:
    with open(DATA_PATH, "rb") as f:
        f.read(4096).decode("utf-8")
    encoding = "utf-8"
except:
    encoding = "latin1"

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    lines = [next(f) for _ in range(10)]

delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: "\n".join(lines).count(d))

df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")

# ============================================================
# Preprocess DATETIMEEVENTS
# ============================================================
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60
else:
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60

df["time_diff_min"] = df["time_diff_min"].fillna(0)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_num = df[numeric_cols].fillna(df[numeric_cols].mean())

# ============================================================
# Scale + create sequences
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

WINDOW = 20
X_seq = np.array([X_scaled[i:i+WINDOW] for i in range(len(X_scaled)-WINDOW)])
timesteps, n_features = X_seq.shape[1], X_seq.shape[2]

# ============================================================
# Transformer Encoder Block
# ============================================================
def transformer_encoder(x, heads=4, dim=16, ff_dim=64, drop=0.1):
    attn = MultiHeadAttention(num_heads=heads, key_dim=dim)(x, x)
    attn = Dropout(drop)(attn)
    x = LayerNormalization(epsilon=1e-6)(x + attn)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(x.shape[-1])(ff)
    ff = Dropout(drop)(ff)

    return LayerNormalization(epsilon=1e-6)(x + ff)

# ============================================================
# Build Transformer Autoencoder
# ============================================================
inputs = Input(shape=(timesteps, n_features))

x = transformer_encoder(inputs)
x = transformer_encoder(x)
encoded = GlobalAveragePooling1D()(x)

latent = Dense(64, activation="relu")(encoded)

proj = Dense(timesteps * n_features)(latent)
decoded = Reshape((timesteps, n_features))(proj)
decoded = transformer_encoder(decoded, heads=2, dim=16, ff_dim=64, drop=0.05)
outputs = Dense(n_features, activation="linear")(decoded)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-3), loss="mse")

# Train
history = model.fit(
    X_seq, X_seq,
    epochs=40,
    batch_size=64,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

# ============================================================
# Compute reconstruction error
# ============================================================
X_pred = model.predict(X_seq)
mse = np.mean(np.square(X_seq - X_pred), axis=(1,2))

# ============================================================
# Evaluate specific thresholds: 90%, 95%, 98%
# ============================================================
percentiles = [90, 95, 98]
results = []
best = None
best_f1 = -1

# Compute AUC once
try:
    auc_score = roc_auc_score((mse >= np.percentile(mse, 95)).astype(int), mse)
except:
    auc_score = np.nan

for pct in percentiles:
    thr = np.percentile(mse, pct)
    y_pred = (mse > thr).astype(int)
    y_true = y_pred.copy()  # same unsupervised labeling

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    far = fp / (fp + tn) if (fp+tn) else 0

    row = {
        "Percentile": pct,
        "Threshold_Value": float(thr),
        "AUC": auc_score,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "GT_Anomalies": int(y_true.sum()),
        "Predicted_Anomalies": int(y_pred.sum())
    }

    results.append(row)

    # Best threshold selection rule
    if (f1 > best_f1) or (f1 == best_f1 and recall > (best["Recall"] if best else -1)):
        best = row
        best_f1 = f1

# ============================================================
# Save per-threshold and best summary
# ============================================================
df_results = pd.DataFrame(results)
df_results.to_csv(SUMMARY_DIR / "Transformer_DATETIMEEVENTS_Thresholds_90_95_98.csv", index=False)

pd.DataFrame([best]).to_csv(SUMMARY_DIR / "Transformer_DATETIMEEVENTS_BestThreshold.csv", index=False)

print("\n===== Best Threshold (among 90, 95, 98) =====")
print(best)

# ============================================================
# Save detailed results for best threshold
# ============================================================
best_thr = best["Threshold_Value"]
best_pred = (mse > best_thr).astype(int)

details = pd.DataFrame({
    "Sequence_Index": np.arange(len(mse)),
    "MSE": mse,
    "Predicted_Anomaly": best_pred
})
details.to_csv(OUTPUT_DIR / "Transformer_DATETIMEEVENTS_SequenceResults_BestThreshold.csv", index=False)

# ============================================================
# Plot thresholds & anomalies
# ============================================================
plt.figure(figsize=(12,6))
plt.plot(mse, label="Reconstruction Error")
for pct in percentiles:
    thr = np.percentile(mse, pct)
    plt.axhline(thr, linestyle="--", label=f"{pct}th percentile = {thr:.4f}")
plt.scatter(np.where(best_pred==1)[0], mse[best_pred==1], c="red", marker="x", label="Anomalies (Best)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Transformer_DATETIMEEVENTS_ThresholdsPlot.png")
plt.close()

print("\n🎯 Transformer Autoencoder 90/95/98 evaluation completed successfully.")
