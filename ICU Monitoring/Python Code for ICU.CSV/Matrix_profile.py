import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)
from stumpy import stump

# =========================
# Configuration / Paths
# =========================
INPUT_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv")
RESULTS_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\Matrix-Profile")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = RESULTS_DIR / "MatrixProfile_Anomaly_Summary.csv"
RESULTS_CSV = RESULTS_DIR / "MatrixProfile_Anomaly_Results.csv"
PLOT_PATH = RESULTS_DIR / "MatrixProfile_Anomaly_Plot.png"

# =========================
# Physiological thresholds (used to create ground-truth labels)
# =========================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90
}

VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR"],
    "SBP": ["SysBP", "SBP", "SystolicBP"]
}


def label_anomalies_from_thresholds(df, features):
    """
    Create a binary ground-truth (1=anomaly, 0=normal) based on simple physiological thresholds.
    Returns a numpy array of length len(df).
    """
    gt = np.zeros(len(df), dtype=int)

    def mark(series, vital_key):
        series = pd.to_numeric(series, errors="coerce").reset_index(drop=True)
        if vital_key == "HR":
            gt_idx = series[(series > THRESHOLDS["HR_tachy"]) | (series < THRESHOLDS["HR_brady"])].index
        elif vital_key == "SBP":
            gt_idx = series[(series > THRESHOLDS["SBP_high"]) | (series < THRESHOLDS["SBP_low"])].index
        else:
            gt_idx = []
        return gt_idx

    for col in features:
        vital = next((v for v, cols in VITAL_MAPPING.items() if col in cols), None)
        if vital is None:
            continue
        try:
            idxs = mark(df[col], vital)
            gt[idxs] = 1
        except Exception:
            # if a column is missing or all NaNs, skip
            continue

    return gt


def evaluate_using_scores(gt, scores, search_percentiles=(90, 99.9), num_steps=100):
    """
    Given ground-truth binary labels (gt) and anomaly 'scores' (higher == more anomalous),
    compute AUC and find the best threshold (by sweeping percentiles) that maximizes F1.
    Returns a dict with AUC, best threshold, F1, Precision, Recall, FAR and the binary prediction.
    """
    valid_mask = ~np.isnan(scores)
    if valid_mask.sum() == 0 or gt.sum() == 0:
        # No valid scores or no anomalies in ground truth -> return NaNs/zeros
        return {
            "AUC": np.nan,
            "Best_Threshold": np.nan,
            "F1": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "FAR": 0.0,
            "Preds": np.zeros_like(scores, dtype=int)
        }

    gt_valid = gt[valid_mask]
    scores_valid = scores[valid_mask]

    try:
        auc = roc_auc_score(gt_valid, scores_valid)
    except Exception:
        auc = np.nan

    # Sweep thresholds by percentile to maximize F1
    best_f1 = -1.0
    best_thresh = np.percentile(scores_valid, search_percentiles[0])
    percentiles = np.linspace(search_percentiles[0], search_percentiles[1], num_steps)
    for p in percentiles:
        thresh = np.percentile(scores_valid, p)
        preds = (scores_valid >= thresh).astype(int)
        if preds.sum() == 0 and gt_valid.sum() == 0:
            f1 = 1.0
        else:
            f1 = f1_score(gt_valid, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_preds_valid = preds.copy()

    # Reconstruct full-length preds array (NaNs -> 0)
    preds_full = np.zeros_like(scores, dtype=int)
    preds_full[valid_mask] = best_preds_valid

    # Compute final metrics on valid region
    tn, fp, fn, tp = confusion_matrix(gt_valid, best_preds_valid).ravel() if best_preds_valid.size and gt_valid.size and (len(np.unique(best_preds_valid))>0 or len(np.unique(gt_valid))>0) else (0,0,0,0)
    precision = precision_score(gt_valid, best_preds_valid, zero_division=0)
    recall = recall_score(gt_valid, best_preds_valid, zero_division=0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "AUC": auc,
        "Best_Threshold": best_thresh,
        "F1": float(best_f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "FAR": float(far),
        "Preds": preds_full
    }


# =========================
# 1) Load & prepare data
# =========================
print("\n=== Matrix Profile anomaly detection (SysBP & Pulse) ===")
df = pd.read_csv(INPUT_PATH)

# Choose features used for detection and ground-truth labeling
FEATURES = ["SysBP", "Pulse"]
df = df[FEATURES].dropna().reset_index(drop=True)

if df.empty:
    raise ValueError(f"No data found in {INPUT_PATH} after selecting features {FEATURES}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Use SysBP for matrix profile analysis (you can switch to other signals)
signal = X_scaled[:, 0]

# =========================
# 2) Compute matrix profile (stump)
# =========================
window_size = 20  # adjust as needed
mp = stump(signal, m=window_size)  # returns array shape (n-m+1, 2); first col is matrix profile
matrix_profile_vals = mp[:, 0]  # length = len(signal) - window_size + 1

# Build a full-length 'score' array aligned with original signal indices:
# We'll set scores[i] = profile[i] for i in [0 .. len(matrix_profile_vals)-1], others = nan
scores = np.full(len(signal), np.nan)
scores[:len(matrix_profile_vals)] = matrix_profile_vals

# Note: matrix_profile higher == more anomalous (discord). If you want inverse, adjust accordingly.

# =========================
# 3) Create anomaly flags from top discords (optional)
# =========================
num_top_discords = 5
discord_indices = np.argsort(matrix_profile_vals)[-num_top_discords:][::-1]  # positions where subsequences are most discordant

# Create a binary anomaly flag array (length = len(signal))
anomaly_flags = np.zeros(len(signal), dtype=int)
for idx in discord_indices:
    start = int(idx)
    end = min(len(signal), start + window_size)
    anomaly_flags[start:end] = 1

# =========================
# 4) Ground-truth labeling using physiological thresholds
# =========================
gt = label_anomalies_from_thresholds(df, FEATURES)  # 0/1 array same length as df

# =========================
# 5) Evaluate using matrix-profile 'scores' (sweep threshold to maximize F1)
# =========================
eval_results = evaluate_using_scores(gt, scores, search_percentiles=(90, 99.9), num_steps=200)

# Also evaluate the fixed 'anomaly_flags' (top-k discords) as a baseline
# Ensure we compare only on valid range (where scores are not nan) for fairness
valid_mask = ~np.isnan(scores)
if valid_mask.sum() > 0:
    gt_valid = gt[valid_mask]
    flags_valid = anomaly_flags[valid_mask]
    try:
        auc_flags = roc_auc_score(gt_valid, scores[valid_mask])  # AUC from scores (not flags)
    except Exception:
        auc_flags = np.nan
    tn, fp, fn, tp = confusion_matrix(gt_valid, flags_valid).ravel() if flags_valid.size and gt_valid.size and (len(np.unique(flags_valid))>0 or len(np.unique(gt_valid))>0) else (0,0,0,0)
    precision_flags = precision_score(gt_valid, flags_valid, zero_division=0)
    recall_flags = recall_score(gt_valid, flags_valid, zero_division=0)
    f1_flags = f1_score(gt_valid, flags_valid, zero_division=0)
    far_flags = fp / (fp + tn) if (fp + tn) > 0 else 0.0
else:
    auc_flags = np.nan
    precision_flags = recall_flags = f1_flags = far_flags = 0.0

# =========================
# 6) Save results (detailed + summary)
# =========================
df_results = pd.DataFrame({
    "Index": np.arange(len(signal)),
    "SysBP": df["SysBP"].values,
    "Pulse": df["Pulse"].values,
    "MatrixProfileScore": scores,
    "TopDiscordFlag": anomaly_flags,
    "GT_Anomaly": gt
})
df_results.to_csv(RESULTS_CSV, index=False)

summary_df = pd.DataFrame([{
    "Feature": "SysBP (matrix profile)",
    "AUC_scores_based": eval_results["AUC"],
    "Best_Threshold": eval_results["Best_Threshold"],
    "F1_scores_based": eval_results["F1"],
    "Precision_scores_based": eval_results["Precision"],
    "Recall_scores_based": eval_results["Recall"],
    "FAR_scores_based": eval_results["FAR"],
    # also include the top-k discord baseline metrics for comparison
    "AUC_topk_discords": auc_flags,
    "F1_topk_discords": f1_flags,
    "Precision_topk_discords": precision_flags,
    "Recall_topk_discords": recall_flags,
    "FAR_topk_discords": far_flags,
    "Notes": f"window_size={window_size}, top_{num_top_discords}_discords"
}])

summary_df.to_csv(SUMMARY_FILE, index=False)

# =========================
# 7) Plot (signal + detected anomalies)
# =========================
plt.figure(figsize=(14, 6))
plt.plot(df["SysBP"].values, label="SysBP (original)", alpha=0.8)
# highlight top-k discord windows
discord_positions = np.where(anomaly_flags == 1)[0]
if discord_positions.size:
    plt.scatter(discord_positions, df["SysBP"].values[discord_positions], color="red", marker="x", s=30, label=f"Top {num_top_discords} discords (window)")
# optionally mark physiological GT anomalies
gt_positions = np.where(gt == 1)[0]
if gt_positions.size:
    plt.scatter(gt_positions, df["SysBP"].values[gt_positions], facecolors='none', edgecolors='green', s=40, label="GT anomalies (thresholds)")

plt.title("Matrix Profile Anomaly Detection (SysBP) — top-k discords vs. GT")
plt.xlabel("Index")
plt.ylabel("SysBP")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
plt.close()

# =========================
# 8) Final printout
# =========================
print("\n=== Evaluation (matrix-profile scores, threshold swept to maximize F1) ===")
print(f"AUC: {eval_results['AUC']}")
print(f"Best Threshold: {eval_results['Best_Threshold']}")
print(f"F1: {eval_results['F1']}")
print(f"Precision: {eval_results['Precision']}")
print(f"Recall: {eval_results['Recall']}")
print(f"FAR: {eval_results['FAR']}")

print("\n=== Baseline (top-k discords) metrics ===")
print(f"AUC (scores): {auc_flags}")
print(f"F1: {f1_flags} | Precision: {precision_flags} | Recall: {recall_flags} | FAR: {far_flags}")

print(f"\nResults saved to:\n - detailed results: {RESULTS_CSV}\n - summary: {SUMMARY_FILE}\n - plot: {PLOT_PATH}")
