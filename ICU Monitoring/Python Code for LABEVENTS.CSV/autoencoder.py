import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ====================== Paths ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\Autoencoder-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Autoencoder Anomaly Detection ===")

# ====================== Detect Encoding & Delimiter ======================
try:
    with open(DATA_PATH, "rb") as f:
        sample = f.read(4096)
    encoding = "utf-8"
    sample.decode(encoding)
except Exception:
    encoding = "latin1"
print(f"Detected encoding: {encoding}")

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    sample_lines = "\n".join([next(f) for _ in range(10)])

delim_candidates = [",", ";", "|", "\t"]
delim_counts = {d: sample_lines.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"Detected delimiter: '{best_delim}'")

# ====================== Load CSV ======================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"Loaded shape: {df.shape} | Columns: {df.columns.tolist()}")

# ====================== Preprocessing ======================
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
else:
    df["time_diff_min"] = np.arange(len(df))

df["time_diff_min"].fillna(0, inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "time_diff_min" not in numeric_cols:
    numeric_cols.append("time_diff_min")

print(f"Using numeric columns: {numeric_cols}")

df_num = df[numeric_cols].copy()
df_num = df_num.apply(pd.to_numeric, errors="coerce")
df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
df_num.dropna(how="all", inplace=True)

if df_num.empty:
    raise ValueError("No valid numeric data found.")

# ====================== Scaling ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# ====================== Autoencoder Model ======================
input_dim = X_scaled.shape[1]

inputs = Input(shape=(input_dim,))
x = Dense(128, activation="relu")(inputs)
x = Dropout(0.2)(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)

x = Dense(64, activation="relu")(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)

outputs = Dense(input_dim, activation="linear")(x)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer=Adam(1e-3), loss="mse")

es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

# Train
history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    shuffle=True,
    callbacks=[es],
    verbose=1
)

# ====================== Reconstruction Error ======================
recons = autoencoder.predict(X_scaled)
mse = np.mean(np.square(X_scaled - recons), axis=1)

# ====================== Ground Truth (no real labels available) ======================
gt = np.zeros(len(mse), dtype=int)

# ====================== MULTI-THRESHOLD EVALUATION ======================
thresholds = [90, 95, 98]
eval_rows = []
best_f1 = -1
best_row = None

for pct in thresholds:
    thr = np.percentile(mse, pct)
    y_pred = (mse >= thr).astype(int)

    # Metrics
    try:
        auc = roc_auc_score(gt, mse)
    except:
        auc = np.nan

    precision = precision_score(gt, y_pred, zero_division=0)
    recall = recall_score(gt, y_pred, zero_division=0)
    f1 = f1_score(gt, y_pred, zero_division=0)

    cm = confusion_matrix(gt, y_pred)
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

# Save CSV files
multi_df = pd.DataFrame(eval_rows)
multi_df.to_csv(SUMMARY_DIR / "LABEVENTS_MultiThresholdSummary.csv", index=False)

pd.DataFrame([best_row]).to_csv(SUMMARY_DIR / "LABEVENTS_BestThreshold.csv", index=False)

print("\n=== MULTI-THRESHOLD EVALUATION ===")
print(multi_df)
print("\n=== BEST THRESHOLD ===")
print(best_row)

# ====================== Apply Best Threshold ======================
best_threshold = best_row["Threshold_Value"]
final_pred = (mse >= best_threshold).astype(int)
anomaly_indices = np.where(final_pred == 1)[0]

# ====================== Save Results ======================
results_df = pd.DataFrame({
    "reconstruction_error": mse,
    "is_anomaly": final_pred
})
results_df.to_csv(OUTPUT_DIR / "LABEVENTS_Autoencoder_Results.csv", index=False)

# ====================== Plot ======================
plt.figure(figsize=(12,6))
plt.plot(mse, label="Reconstruction Error", alpha=0.7)
plt.axhline(best_threshold, color='orange', linestyle='--', label=f"Best Threshold ({best_row['Threshold_%']}%)")
plt.scatter(anomaly_indices, mse[anomaly_indices], color='red', marker='x', label="Anomalies")
plt.title("Autoencoder Anomaly Detection - LABEVENTS")
plt.xlabel("Record Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_Autoencoder_AnomalyPlot.png")
plt.close()

print("\n🎯 Autoencoder anomaly detection completed successfully!")
print("📌 Multi-threshold summary saved.")
print("📌 Best threshold saved.")
