import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\Autoencoder-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Autoencoder-based Anomaly Detection (updated) ===")

# -------------------------
# Load CSV robustly
# -------------------------
encoding = "utf-8"
try:
    with open(DATA_PATH, "rb") as f:
        _ = f.read(4096).decode(encoding)
except Exception:
    encoding = "latin1"

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    sample_lines = [next(f) for _ in range(20)]
sample_text = "\n".join(sample_lines)
delim_candidates = [",", ";", "\t", "|"]
delim = max(delim_candidates, key=lambda d: sample_text.count(d))

df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=delim, engine="python", on_bad_lines="skip")
print(f"Loaded file: shape = {df.shape} | detected delimiter = '{delim}' | encoding = {encoding}")

# -------------------------
# Feature engineering (unchanged)
# -------------------------
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df = df.sort_values(["subject_id", "charttime"]).reset_index(drop=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
    event_counts = df.groupby("subject_id").size().rename("event_count")
    mean_timediff = df.groupby("subject_id")["time_diff_min"].mean().rename("avg_time_gap_min")
    df_features = pd.concat([event_counts, mean_timediff], axis=1).reset_index()
else:
    if "charttime" in df.columns:
        df = df.sort_values("charttime").reset_index(drop=True)
        df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60.0
        df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
        df_features = pd.DataFrame({
            "event_count": [len(df)],
            "avg_time_gap_min": [df["time_diff_min"].mean()]
        })
    else:
        df_features = pd.DataFrame({
            "event_count": [len(df)],
            "avg_time_gap_min": [0.0]
        })

df_features = df_features.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)
print(f"Derived features: {df_features.columns.tolist()} | records = {len(df_features)}")

# -------------------------
# Rule-based ground-truth (unchanged)
# -------------------------
y_true = np.zeros(len(df_features), dtype=int)
for col in df_features.columns:
    series = df_features[col]
    mean, std = series.mean(), series.std(ddof=0)
    if std == 0 or np.isnan(std):
        continue
    upper_thr = mean + 2 * std
    anomaly_mask = series > upper_thr
    y_true[anomaly_mask.values] = 1

print(f"GT anomalies detected = {y_true.sum()}")

# -------------------------
# Train Autoencoder (unchanged)
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features.values)

input_dim = X_scaled.shape[1]
encoding_dim = max(2, input_dim // 2)

inp = Input(shape=(input_dim,))
x = Dense(encoding_dim * 2, activation="relu")(inp)
x = Dropout(0.1)(x)
encoded = Dense(encoding_dim, activation="relu")(x)
x = Dense(encoding_dim * 2, activation="relu")(encoded)
x = Dropout(0.1)(x)
out = Dense(input_dim, activation="linear")(x)

autoencoder = Model(inputs=inp, outputs=out)
autoencoder.compile(optimizer=Adam(1e-3), loss="mse")

early_stop = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

autoencoder.fit(
    X_scaled, X_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.1 if len(X_scaled) > 10 else 0.0,
    callbacks=[early_stop],
    verbose=0
)

reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.square(X_scaled - reconstructions), axis=1)

# ================================================================
#      ** UPDATED SECTION: 90 / 95 / 98 percentile thresholds **
# ================================================================
thresholds = {
    "90%": np.percentile(mse, 90),
    "95%": np.percentile(mse, 95),
    "98%": np.percentile(mse, 98)
}

all_results = []
best_result = None

for name, thr in thresholds.items():
    y_pred = (mse > thr).astype(int)

    try:
        auc = roc_auc_score(y_true, mse)
    except:
        auc = np.nan

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    res = {
        "Threshold": name,
        "Threshold_Value": float(thr),
        "AUC": float(auc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "FAR": float(far)
    }

    all_results.append(res)

    if best_result is None or f1 > best_result["F1"]:
        best_result = res

print("\n===== MULTI-THRESHOLD RESULTS =====")
for r in all_results:
    print(r)

print("\n===== BEST THRESHOLD BASED ON F1 =====")
print(best_result)

# -------------------------
# Save summary CSV (multi-threshold)
# -------------------------
summary_df = pd.DataFrame(all_results)
summary_df["Model"] = "Autoencoder_DATETIMEEVENTS"

summary_file = SUMMARY_DIR / "Autoencoder_DATETIMEEVENTS_MultiThreshold_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Saved summary → {summary_file}")

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(12,6))
plt.plot(mse, label="MSE Reconstruction Error")
plt.axhline(thresholds["95%"], color="orange", linestyle="--", label="95% Threshold")

plt.title("Reconstruction Error Plot — DATETIMEEVENTS Autoencoder")
plt.xlabel("Index")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)

plot_file = OUTPUT_DIR / "DATETIMEEVENTS_AE_MSE_Plot.png"
plt.savefig(plot_file, dpi=300)
plt.close()

print(f"Saved plot → {plot_file}")
print("\n✅ Completed successfully.")
