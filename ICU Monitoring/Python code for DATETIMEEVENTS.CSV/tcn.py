import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# Paths
# =========================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\TCN")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

print("\n=== Running TCN Anomaly Detection with 90/95/98 threshold evaluation ===")

# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH, engine="python", on_bad_lines="skip")

# Convert charttime
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

# =========================
# Create time_diff_min
# =========================
if "subject_id" in df.columns and "charttime" in df.columns and df["charttime"].notna().sum() > 1:
    df = df.sort_values(["subject_id", "charttime"])
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60
else:
    df = df.sort_values("charttime") if "charttime" in df.columns else df
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60 if "charttime" in df.columns else np.arange(len(df))

df["time_diff_min"] = df["time_diff_min"].fillna(0)

# =========================
# Select numeric columns
# =========================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df[num_cols].copy()
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
df_numeric.fillna(df_numeric.mean(), inplace=True)

print(f"Using numeric columns: {num_cols}")

# =========================
# True anomaly labels (SysBP / Pulse) - same as your pipeline
# =========================
gt = np.zeros(len(df_numeric), dtype=int)

if "SysBP" in df_numeric.columns:
    sbp = df_numeric["SysBP"].values
    gt = np.where((sbp > 140) | (sbp < 90), 1, gt)

if "Pulse" in df_numeric.columns:
    pulse = df_numeric["Pulse"].values
    gt = np.where((pulse > 100) | (pulse < 60), 1, gt)

# =========================
# Scale and create sequences
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

SEQ_LEN = 10

def create_sequences(X, length):
    return np.array([X[i:i+length] for i in range(len(X) - length)])

X_seq = create_sequences(X_scaled, SEQ_LEN)

# Label a sequence anomalous if ANY point inside is anomalous
y_true_seq = np.array([1 if gt[i:i+SEQ_LEN].sum() > 0 else 0 for i in range(len(X_seq))])

print(f"Sequences: {X_seq.shape}, True anomalies in sequences: {y_true_seq.sum()}")

# =========================
# TCN Autoencoder Model (unchanged architecture)
# =========================
inputs = Input(shape=(SEQ_LEN, X_seq.shape[2]))
x = Conv1D(64, 3, activation="relu", padding="causal")(inputs)
x = Dropout(0.1)(x)
x = Conv1D(32, 3, activation="relu", padding="causal")(x)
x = Flatten()(x)
encoded = Dense(32, activation="relu")(x)

x = Dense(SEQ_LEN * X_seq.shape[2], activation="relu")(encoded)
x = Reshape((SEQ_LEN, X_seq.shape[2]))(x)
x = Conv1D(32, 3, activation="relu", padding="causal")(x)
decoded = Conv1D(X_seq.shape[2], 3, activation="linear", padding="same")(x)

model = Model(inputs, decoded)
model.compile(optimizer=Adam(1e-3), loss="mse")

model.fit(
    X_seq, X_seq,
    epochs=40,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

# =========================
# Reconstruction Error
# =========================
recon = model.predict(X_seq)
mse = np.mean(np.square(X_seq - recon), axis=(1, 2))

# =========================
# Evaluate exactly at 90%, 95%, 98% thresholds
# =========================
percentiles = [90, 95, 98]
rows = []
best = None
best_f1 = -1.0

# compute AUC once (uses continuous scores)
try:
    auc_score = roc_auc_score(y_true_seq, mse) if len(np.unique(y_true_seq)) > 1 else np.nan
except Exception:
    auc_score = np.nan

for pct in percentiles:
    thr = np.percentile(mse, pct)
    y_pred = (mse >= thr).astype(int)

    # metrics (handle degenerate CM shapes)
    try:
        precision = precision_score(y_true_seq, y_pred, zero_division=0)
        recall = recall_score(y_true_seq, y_pred, zero_division=0)
        f1 = f1_score(y_true_seq, y_pred, zero_division=0)
    except Exception:
        precision = recall = f1 = 0.0

    cm = confusion_matrix(y_true_seq, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # fallback if only one class present in y_true or y_pred
        tn = fp = fn = tp = 0
        # try to infer counts if possible
        if cm.shape == (1,1):
            if y_true_seq[0] == 0 and y_pred.sum() == 0:
                tn = cm[0,0]
            elif y_true_seq[0] == 1 and y_pred.sum() == 1:
                tp = cm[0,0]

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    row = {
        "Percentile": pct,
        "Threshold_Value": float(thr),
        "AUC(of MSE)": float(auc_score) if not np.isnan(auc_score) else np.nan,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "FAR": float(far),
        "GT_Anomalies": int(y_true_seq.sum()),
        "Predicted_Anomalies": int(y_pred.sum())
    }
    rows.append(row)

    # choose best among these three: primary F1, tie-breaker recall then precision
    if (f1 > best_f1) or (f1 == best_f1 and (recall > (best.get("Recall") if best else -1))):
        best_f1 = f1
        best = row

# =========================
# Save per-threshold CSV and best-threshold CSV
# =========================
thresholds_df = pd.DataFrame(rows)
thresholds_file = SUMMARY_DIR / "TCN_DATETIMEEVENTS_Thresholds_90_95_98.csv"
thresholds_df.to_csv(thresholds_file, index=False)
print(f"\nSaved per-threshold metrics → {thresholds_file}")

best_file = SUMMARY_DIR / "TCN_DATETIMEEVENTS_BestThreshold.csv"
pd.DataFrame([best]).to_csv(best_file, index=False)
print(f"Saved best-threshold summary → {best_file}")

print("\nAll evaluated thresholds (90,95,98):")
print(thresholds_df)
print("\nBest threshold chosen among them:")
print(best)

# =========================
# Save sequence-level predictions for the best threshold (optional)
# =========================
best_thr = best["Threshold_Value"]
best_y_pred = (mse >= best_thr).astype(int)
results_df = pd.DataFrame({
    "Sequence_Index": np.arange(len(mse)),
    "Reconstruction_MSE": mse,
    "Pred_Label_at_BestThreshold": best_y_pred,
    "GT_Label": y_true_seq
})
results_csv = OUTPUT_DIR / "TCN_DATETIMEEVENTS_Sequence_Results_BestThreshold.csv"
results_df.to_csv(results_csv, index=False)
print(f"Saved sequence-level results → {results_csv}")

# =========================
# Plot reconstruction error & mark thresholds
# =========================
plt.figure(figsize=(12,6))
plt.plot(mse, label="Reconstruction MSE")
for r in rows:
    plt.axhline(r["Threshold_Value"], linestyle="--", label=f"{int(r['Percentile'])}th pct = {r['Threshold_Value']:.6f}")
# mark predicted anomalies for best threshold
pred_centers = np.where(best_y_pred == 1)[0]
plt.scatter(pred_centers, mse[pred_centers], color="red", marker="x", label="Predicted anomalies (best)")
plt.title("TCN Reconstruction Error — thresholds 90/95/98 (best marked)")
plt.xlabel("Sequence index")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "TCN_DATETIMEEVENTS_MSE_with_thresholds.png")
plt.close()
print(f"Saved plot → {OUTPUT_DIR / 'TCN_DATETIMEEVENTS_MSE_with_thresholds.png'}")

print("\n✅ TCN anomaly detection completed with 90/95/98 evaluation. Best threshold chosen only from those three.")
