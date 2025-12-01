import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =========================
# Paths
# =========================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\OneClassSVM-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing DATETIMEEVENTS.csv for One-Class SVM Anomaly Detection (90/95/98 pct eval) ===")

# =========================
# Encoding + delimiter detection
# =========================
try:
    with open(DATA_PATH, 'rb') as f:
        f.read(4096).decode("utf-8")
    encoding = "utf-8"
except:
    encoding = "latin1"
print(f"Detected encoding: {encoding}")

with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    line = f.readline()
delim = "," if line.count(",") >= line.count(";") else ";"
print(f"Detected delimiter: '{delim}'")

# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=delim, engine="python", on_bad_lines="skip")
print(f"Loaded data shape: {df.shape}")

# =========================
# Preprocessing for time-based features
# =========================
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60
else:
    # if charttime not present or sorting fails, produce NaNs which we will impute later
    if "charttime" in df.columns:
        df.sort_values("charttime", inplace=True)
        df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60
    else:
        # fallback synthetic
        df["time_diff_min"] = np.arange(len(df)).astype(float)

# Replace initial NaN time diffs with 0 so imputer doesn't misinterpret too many NaNs
df["time_diff_min"] = df["time_diff_min"].fillna(0)

# =========================
# Select numeric data
# =========================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df[numeric_cols].copy()
print(f"Initial numeric features ({len(numeric_cols)}): {numeric_cols}")

# =========================
# Drop obvious identifier-like columns (keep meaningful numeric features)
# =========================
# This preserves pipeline behavior but removes noisy ID columns that don't help model
id_patterns = ["id", "row", "itemid", "cgid", "hadm_id", "icustay_id"]
cols_to_drop = [c for c in df_numeric.columns if any(p in c.lower() for p in id_patterns)]
if cols_to_drop:
    print(f"Dropping identifier-like columns: {cols_to_drop}")
    df_numeric = df_numeric.drop(columns=cols_to_drop)

# =========================
# Replace inf and -inf then impute remaining NaNs with column mean
# =========================
df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove constant columns (zero variance) because they don't help scaling or SVM
const_cols = [c for c in df_numeric.columns if df_numeric[c].nunique(dropna=True) <= 1]
if const_cols:
    print(f"Dropping constant columns: {const_cols}")
    df_numeric = df_numeric.drop(columns=const_cols)

# If after dropping we have no columns, fall back to keeping 'time_diff_min' if present
if df_numeric.shape[1] == 0:
    if "time_diff_min" in df.columns:
        df_numeric = df[["time_diff_min"]].copy()
        print("Fell back to using only 'time_diff_min' as the feature.")
    else:
        raise ValueError("No usable numeric columns remain after dropping identifiers and constants.")

# Impute missing values with column mean
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(df_numeric)

# Sanity check: no NaNs now
if np.isnan(X_imputed).any():
    raise ValueError("Imputation failed: NaNs remain in feature matrix.")

print(f"Final feature set used for modeling: {list(df_numeric.columns)}")
print(f"Final numeric dataset shape: {X_imputed.shape}")

# =========================
# Scale the data
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# =========================
# One-Class SVM Training
# =========================
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
svm.fit(X_scaled)

raw_pred = svm.predict(X_scaled)                # -1 anomaly, 1 normal
decision_scores = -svm.decision_function(X_scaled)  # higher -> more anomalous
pred_labels = np.where(raw_pred == -1, 1, 0)    # 1 = anomaly

# =========================
# Multi-threshold evaluation (only 90,95,98) — best chosen from these
# =========================
thresholds = [90, 95, 98]
eval_rows = []
best_row = None
best_f1 = -1.0

for pct in thresholds:
    thr_val = np.percentile(decision_scores, pct)

    # Build ground truth using this percentile (as you requested)
    y_true = (decision_scores >= thr_val).astype(int)   # 1 = GT anomaly

    # Use the model's predicted labels as y_pred (unchanged pipeline)
    y_pred = pred_labels

    # Compute metrics; guard for constant labels
    try:
        auc = roc_auc_score(y_true, decision_scores) if (len(np.unique(y_true)) > 1) else np.nan
    except Exception:
        auc = np.nan

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    # Robust unpacking of confusion matrix
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # possible shapes if one class missing
        tn = fp = fn = tp = 0
        if cm.shape == (1,1):
            if y_true[0] == 0 and y_pred.sum() == 0:
                tn = cm[0,0]
            elif y_true[0] == 1 and y_pred.sum() == 1:
                tp = cm[0,0]
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    row = {
        "Threshold (%)": pct,
        "Threshold Value": float(thr_val),
        "AUC": float(auc) if not np.isnan(auc) else np.nan,
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(f1),
        "FAR": float(far),
        "GT_Anomalies": int(y_true.sum()),
        "Predicted_Anomalies": int(y_pred.sum())
    }
    eval_rows.append(row)

    # choose best by F1 (tie-break: higher recall then precision)
    if f1 > best_f1 or (f1 == best_f1 and r > (best_row.get("Recall") if best_row else -1)):
        best_f1 = f1
        best_row = row

# =========================
# Report / Save
# =========================
print("\nThreshold evaluations (90/95/98):")
for r in eval_rows:
    print(r)

print("\nBEST threshold among [90,95,98] (by F1):")
print(best_row)

eval_df = pd.DataFrame(eval_rows)
eval_file = SUMMARY_DIR / "OneClassSVM_DATETIMEEVENTS_Threshold_Evaluation.csv"
eval_df.to_csv(eval_file, index=False)
print(f"\nThreshold evaluation saved → {eval_file}")

best_file = SUMMARY_DIR / "OneClassSVM_DATETIMEEVENTS_BestThreshold.csv"
pd.DataFrame([best_row]).to_csv(best_file, index=False)
print(f"Best threshold summary saved → {best_file}")

# =========================
# Save detailed results
# =========================
results_df = pd.DataFrame({
    "Index": np.arange(X_scaled.shape[0]),
    "Decision_Score": decision_scores,
    "Predicted_Label": pred_labels
})
results_df.to_csv(OUTPUT_DIR / "OneClassSVM_DATETIMEEVENTS_Results.csv", index=False)
print(f"Detailed results saved → {OUTPUT_DIR / 'OneClassSVM_DATETIMEEVENTS_Results.csv'}")

# =========================
# Plot decision scores and best threshold
# =========================
plt.figure(figsize=(12,6))
plt.plot(decision_scores, label="Decision Score")
plt.axhline(best_row["Threshold Value"], color="orange", linestyle="--", label=f"Best Threshold ({best_row['Threshold (%)']}%)")
plt.scatter(np.where(pred_labels == 1), decision_scores[pred_labels == 1], color="red", marker="x", label="Predicted Anomaly")
plt.xlabel("Index")
plt.ylabel("Decision Score")
plt.title("One-Class SVM — DATETIMEEVENTS (90/95/98 threshold evaluation)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DATETIMEEVENTS_OneClassSVM_BestThresholdPlot.png")
plt.close()
print(f"Plot saved → {OUTPUT_DIR / 'DATETIMEEVENTS_OneClassSVM_BestThresholdPlot.png'}")

print("\n✅ One-Class SVM anomaly detection completed successfully!")
