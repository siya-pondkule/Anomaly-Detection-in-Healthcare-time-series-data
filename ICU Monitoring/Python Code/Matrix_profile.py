import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import stumpy # Requires: pip install stumpy
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------- Config ----------
DATASETS_DIR = Path(r"D:/Final Year/Project/Anomaly Detection/ICU Monitoring/Datasets")
OUTPUT_DIR = DATASETS_DIR / "../Results of all Models/MatrixProfile-ModelResult"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TH = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90,
    "DBP_high": 90, "DBP_low": 60,
    "MAP_high": 110, "MAP_low": 70,
    "RR_tachypnea": 24, "RR_apnea": 8,
    "SpO2_low": 90, "Temp_high": 38.5, "Temp_low": 35.0
}

VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR", "PR"],
    "SBP": ["SysBP", "SBP", "SystolicBP", "SYS"],
    "DBP": ["DiaBP", "DBP", "DiastolicBP", "DIA"],
    "MAP": ["MAP", "MeanBP", "MeanArterialPressure"],
    "RR": ["RespRate", "RR", "Resp", "RespiratoryRate"],
    "SpO2": ["SpO2", "OxygenSaturation", "O2Sat", "SaO2"],
    "Temp": ["Temp", "Temperature", "BodyTemp", "T"]
}

# ======================
# Detect and LABEL anomalies (Ground Truth)
# ======================
def label_anomalies(df, gt_array):
    """Detects physiological anomalies and updates the Ground Truth array."""
    anomalies_for_cleaning = []
    
    for col_name in df.columns:
        temp_series = pd.to_numeric(df[col_name], errors="coerce").dropna()
        original_indices = temp_series.index
        
        if temp_series.empty:
            continue

        vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
        anomalous_indices_in_df = []

        # --- ANOMALY DETECTION LOGIC (based on TH) ---
        if vital:
            if vital == "HR":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["HR_tachy"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["HR_brady"]])
            elif vital == "SBP":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["SBP_high"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["SBP_low"]])
            elif vital == "DBP":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["DBP_high"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["DBP_low"]])
            elif vital == "MAP":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["MAP_high"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["MAP_low"]])
            elif vital == "RR":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["RR_tachypnea"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["RR_apnea"]])
            elif vital == "SpO2":
                anomalous_indices_in_df += list(original_indices[temp_series < TH["SpO2_low"]])
            elif vital == "Temp":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["Temp_high"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["Temp_low"]])
        else:
            # generic z-score detection
            z = (temp_series - temp_series.mean()) / temp_series.std(ddof=0)
            anomalous_indices_in_df += list(original_indices[np.abs(z) > 3])

        # Update the Ground Truth array (1 for anomaly)
        for idx in anomalous_indices_in_df:
            if idx < len(gt_array):
                gt_array[idx] = 1
                anomalies_for_cleaning.append({"vital": col_name, "index": idx})
                
    return anomalies_for_cleaning

# ======================
# Performance Metrics Calculation
# ======================
def evaluate_anomaly_detection(y_true, anomaly_scores, file_name, dataset_name):
    """
    Calculates all required performance metrics (AUC, F1, P, R, FAR) 
    using the Matrix Profile scores (higher = more anomalous).
    """
    
    scores = anomaly_scores # Matrix Profile distance is the score (higher = anomaly)
    
    # 1. Check for valid ground truth
    if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return {
            "File": file_name, "Dataset": dataset_name, "Model": "Matrix Profile",
            "AUC": np.nan, "F1": 0.0, "Precision": 0.0, "Recall": 0.0, 
            "False Alarm Rate (FAR)": 0.05, "Optimal Threshold": np.nan, "GT Anomaly Count": np.sum(y_true)
        }

    # 2. AUC 
    try:
        if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
            auc = roc_auc_score(y_true, scores)
        else:
            auc = np.nan
    except ValueError:
        auc = np.nan

    # 3. Optimal Threshold Search (Maximize F1 Score)
    best_f1 = 0.0
    # Search threshold candidates across the upper tail of the scores
    threshold_candidates = np.linspace(np.percentile(scores, 90), np.max(scores), 50)
    best_threshold = np.percentile(scores, 95) 

    for threshold in threshold_candidates:
        # Prediction: scores > threshold means anomaly (1)
        y_pred = (scores > threshold).astype(int) 
        
        # Check if F1 can be computed
        if np.sum(y_pred) > 0:
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
    
    # Use the best threshold for final metrics
    y_pred_best = (scores > best_threshold).astype(int)
    
    # 4. Precision, Recall, F1, False Alarm Rate (FAR)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    precision = precision_score(y_true, y_pred_best, zero_division=0)
    recall = recall_score(y_true, y_pred_best, zero_division=0)
    f1 = best_f1
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "File": file_name,
        "Dataset": dataset_name,
        "Model": "Matrix Profile",
        "AUC": auc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "False Alarm Rate (FAR)": false_alarm_rate,
        "Optimal Threshold": best_threshold,
        "GT Anomaly Count": np.sum(y_true)
    }

# ======================
# Matrix Profile Calculation Function - MODIFIED
# ======================
def calculate_matrix_profile(df_clean, window_size):
    """
    Calculates the Matrix Profile (MP) for each column and aggregates the discord scores.
    """
    mp_scores_per_col = {}
    
    for col in df_clean.columns:
        # --- FIX 1: Explicitly convert to float64 for STUMPY ---
        T = df_clean[col].values.astype(np.float64) 
        
        # --- FIX 2: Use ignore_trivial=True to address the Stumpy warning (recommended for self-joins) ---
        mp = stumpy.stump(T, m=window_size, normalize=True, ignore_trivial=True)
        
        # The MP distance array (discord scores) is mp[:, 0]
        n_fill = len(T) - len(mp)
        mp_dist = mp[:, 0]
        mp_padded = np.concatenate((np.full(n_fill, np.nan), mp_dist))
        
        mp_scores_per_col[col] = mp_padded
        
    # Aggregate scores: Take the maximum discord score across all dimensions for each time step.
    mp_df = pd.DataFrame(mp_scores_per_col)
    
    # Drop rows with NaNs (padding)
    mp_df = mp_df.dropna().reset_index(drop=True)
    
    return mp_df.max(axis=1).values, len(df_clean) - len(mp_df)

# ======================
# Plotting Function (for visualization and saving)
# ======================
def plot_matrix_profile_error(anomaly_scores, optimal_threshold, file_stem):
    """Plots the Matrix Profile Discord Score and the optimal F1 threshold."""
    plt.figure(figsize=(10,5))
    plt.plot(anomaly_scores, label="Matrix Profile Discord Score")
    plt.axhline(y=optimal_threshold, color='r', linestyle='--', label='Optimal F1 Threshold')
    plt.title(f"Matrix Profile Discord Scores - {file_stem}")
    plt.xlabel("Subsequence Start Index")
    plt.ylabel("Z-Normalized Euclidean Distance")
    plt.legend()
    plt.tight_layout()
    
    # --- FIX 3: Explicitly save the figure ---
    plot_path = OUTPUT_DIR / f"{file_stem}_mp_error_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Plot saved to {plot_path}")


# ======================
# Main Processing Loop - MODIFIED TO INCLUDE PLOTTING CALL
# ======================
evaluation_summary = []
WINDOW_SIZE = 10 # 10 time steps (hyperparameter for MP)

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} (Matrix Profile) ===")
    try:
        df_original = pd.read_csv(file, low_memory=False)
        df = df_original.copy()
        
        # 1. Data Cleaning and Index Reset
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').reset_index(drop=True)
        if df.empty:
            print("No numeric columns.")
            continue

        # 2. --- GROUND TRUTH GENERATION (BEFORE CLEANING) ---
        gt_array = np.zeros(len(df))
        anomalies_for_cleaning = label_anomalies(df, gt_array)
        y_true_full = gt_array

        # 3. Data Cleaning (to train on 'normal' data)
        df_clean = df.copy()
        for a in anomalies_for_cleaning:
            if a['vital'] in df_clean.columns:
                df_clean.loc[a['index'], a['vital']] = np.nan
                
        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.loc[:, df_clean.nunique() > 1].fillna(df_clean.mean())
        
        if len(df_clean) < WINDOW_SIZE * 2:
             print(f"Skipped: Not enough data for Matrix Profile (needs > {WINDOW_SIZE * 2} rows).")
             continue

        # 4. Matrix Profile Calculation (Anomaly Scores)
        anomaly_scores, padding_count = calculate_matrix_profile(df_clean, WINDOW_SIZE)
        
        # Align Ground Truth with Matrix Profile output (remove padded rows)
        y_true = y_true_full[padding_count:]

        # 5. Evaluation against Ground Truth
        dataset_name = file.stem
        eval_results = evaluate_anomaly_detection(y_true, anomaly_scores, file.name, dataset_name)
        evaluation_summary.append(eval_results)

        print(f"Metrics (Max F1 Threshold): AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | P={eval_results['Precision']:.4f} | R={eval_results['Recall']:.4f} | FAR={eval_results['False Alarm Rate (FAR)']:.4f}")
        print(f"GT Anomalies (Used for Eval): {y_true.sum()} | Total Anomalies Detected (Max F1 Thresh): {np.sum(anomaly_scores > eval_results['Optimal Threshold'])}")
        
        # 6. Plot the discord scores and save the figure
        plot_matrix_profile_error(anomaly_scores, eval_results['Optimal Threshold'], file.stem)

    except Exception as e:
        print(f"❌ CRITICAL ERROR processing {file.name}: {e}")


# ======================
# Save Overall Summary
# ======================
if evaluation_summary:
    eval_df = pd.DataFrame(evaluation_summary)
    eval_df = eval_df.sort_values(by="Dataset").reset_index(drop=True)
    
    summary_path = OUTPUT_DIR / "matrixprofile_performance_summary.csv"
    eval_df.to_csv(summary_path, index=False)
    print(f"\n✅ Matrix Profile Performance Summary saved → {summary_path}")
    print("\nSummary of Performance Metrics (Matrix Profile):")
    print(eval_df[["Dataset", "AUC", "F1", "Precision", "Recall", "False Alarm Rate (FAR)", "GT Anomaly Count"]])

else:
    print("⚠️ No results to summarize.")