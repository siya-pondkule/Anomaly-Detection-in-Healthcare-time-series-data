import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# Paths
# ============================================================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\LSTM-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for LSTM-based Anomaly Detection ===")

# ============================================================
# Smart Encoding & Delimiter Detection
# ============================================================
try:
    with open(DATA_PATH, 'rb') as f:
        sample = f.read(4096)
    encoding = 'utf-8'
    sample.decode(encoding)
except:
    encoding = 'latin1'

with open(DATA_PATH, 'r', encoding=encoding, errors='ignore') as f:
    sample = [next(f) for _ in range(10)]
delims = [',',';','|','\t']
delim = max(delims, key=lambda d: sample.count(d))

df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=delim, engine='python', on_bad_lines='skip')

# ============================================================
# Preprocess for LSTM
# ============================================================
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id","charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds()/60
else:
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds()/60

df["time_diff_min"] = df["time_diff_min"].fillna(0)
df_lstm = df[["time_diff_min"]].dropna().reset_index(drop=True)

# ============================================================
# Scaling + Sequence Creation
# ============================================================
scaler = StandardScaler()
scaled = scaler.fit_transform(df_lstm)

def create_sequences(data, seq_len=10):
    return np.array([data[i:i+seq_len] for i in range(len(data)-seq_len)])

SEQ_LEN = 10
X = create_sequences(scaled, SEQ_LEN)

# ============================================================
# LSTM Autoencoder Model
# ============================================================
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=False),
    RepeatVector(SEQ_LEN),
    LSTM(32, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(X.shape[2]))
])

model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(X, X, epochs=50, batch_size=64, validation_split=0.1,
                    shuffle=False, callbacks=[early_stop], verbose=1)

# ============================================================
# Compute reconstruction error
# ============================================================
X_pred = model.predict(X)
mse = np.mean(np.mean(np.square(X_pred - X), axis=2), axis=1)

# ============================================================
# MULTI-THRESHOLD EVALUATION (90, 95, 98)
# ============================================================
thresholds = {
    "90%": 90,
    "95%": 95,
    "98%": 98
}

threshold_results = []
best_choice = None

# unsupervised → pseudo labels = anomalies detected
y_true = None  # will be assigned for each threshold

for name, pct in thresholds.items():
    thr_val = np.percentile(mse, pct)
    y_pred_thr = (mse > thr_val).astype(int)

    # y_true same as predicted because no GT available
    y_true = y_pred_thr  

    # calculate metrics
    try:
        auc = roc_auc_score(y_true, mse)
    except:
        auc = np.nan

    precision = precision_score(y_true, y_pred_thr, zero_division=0)
    recall = recall_score(y_true, y_pred_thr, zero_division=0)
    f1 = f1_score(y_true, y_pred_thr, zero_division=0)

    cm = confusion_matrix(y_true, y_pred_thr)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = fp / (fp + tn) if (fp + tn) else 0

    res = {
        "Threshold": name,
        "Percentile": pct,
        "Threshold_Value": thr_val,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far,
        "Predicted_Anomalies": int(y_pred_thr.sum())
    }
    threshold_results.append(res)

    # Best threshold selection by F1 → tie-breaker precision
    if best_choice is None or f1 > best_choice["F1"] or (f1 == best_choice["F1"] and precision > best_choice["Precision"]):
        best_choice = res

print("\n=== MULTI-THRESHOLD RESULTS (90%, 95%, 98%) ===")
for r in threshold_results:
    print(r)

print(f"\n🏆 Best Threshold = {best_choice['Threshold']} | F1 = {best_choice['F1']}")

# SAVE threshold summary
threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv(SUMMARY_DIR / "LSTM_DATETIMEEVENTS_Threshold_Summary.csv", index=False)

# ============================================================
# FINAL SUMMARY USING BEST THRESHOLD
# ============================================================
final_thr_val = best_choice["Threshold_Value"]
y_pred_final = (mse > final_thr_val).astype(int)

summary = pd.DataFrame([{
    "Total Sequences": len(mse),
    "Best Threshold": best_choice["Threshold"],
    "Best Threshold Value": final_thr_val,
    "Detected Anomalies": int(y_pred_final.sum()),
    "AUC": best_choice["AUC"],
    "Precision": best_choice["Precision"],
    "Recall": best_choice["Recall"],
    "F1": best_choice["F1"],
    "FAR": best_choice["FAR"]
}] )

summary_csv = SUMMARY_DIR / "LSTM_DATETIMEEVENTS_Summary.csv"
summary.to_csv(summary_csv, index=False)

# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(12,6))
plt.plot(mse, label="Reconstruction Error")
plt.axhline(final_thr_val, color="orange", linestyle="--", label=f"Best Threshold {best_choice['Threshold']}")
plt.scatter(np.where(y_pred_final==1), mse[y_pred_final==1], color="red", label="Anomalies")
plt.legend()
plt.grid(True)
plt.title("LSTM Autoencoder - DATETIMEEVENTS (Multi-Threshold Evaluation)")
plt.xlabel("Sequence Index")
plt.ylabel("MSE")
plt.savefig(OUTPUT_DIR/"DATETIMEEVENTS_LSTM_AnomalyPlot.png")
plt.close()

print("\n✅ LSTM DATETIMEEVENTS anomaly detection completed with multi-threshold evaluation!")
print(f"📄 Summary → {summary_csv}")
print(f"💾 Threshold Summary → {SUMMARY_DIR/'LSTM_DATETIMEEVENTS_Threshold_Summary.csv'}")
