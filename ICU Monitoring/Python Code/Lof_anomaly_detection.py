import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ======================
# Paths & Config
# ======================
# NOTE: The provided path is specific to your local machine.
DATASETS_DIR = Path(r"D:/Final Year/Project/Anomaly Detection/ICU Monitoring/Datasets")
OUTPUT_DIR = DATASETS_DIR / "../Results of all Models/LOF-ModelResults"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)



# ======================
# Thresholds and Vital Sign Mappings (TH, VITAL_MAPPING remains the same)
# ======================
TH = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90,
    "DBP_high": 90, "DBP_low": 60,
    "MAP_high": 110, "MAP_low": 70,
    "RR_tachypnea": 24, "RR_apnea": 8,
    "SpO2_low": 90,
    "Temp_high": 38.5, "Temp_low": 35.0
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
        # Use a temporary series for thresholding that handles NaNs
        series = pd.to_numeric(df[col_name], errors="coerce").dropna().reset_index(drop=True)
        # Note: The original index must be used to update the main gt_array and df_clean.
        # This requires careful index handling. We'll use the original dataframe index.
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
        # Assuming the DF's index is a simple 0-based range after initial cleanup
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
    using the anomaly scores (higher = more anomalous for -LOF).
    """
    
    # LOF returns the Negative Outlier Factor (NOF), where small values are inliers
    # and large absolute values are outliers. We use the absolute value for scores.
    scores = np.abs(anomaly_scores) 
    
    # Check for empty ground truth
    if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return {
            "File": file_name, "Dataset": dataset_name, "Model": "LOF",
            "AUC": np.nan, "F1": 0.0, "Precision": 0.0, "Recall": 0.0, 
            "False Alarm Rate (FAR)": 0.05, "Optimal Threshold": np.nan, "GT Anomaly Count": np.sum(y_true)
        }

    # 1. AUC (requires scores and true labels)
    try:
        if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
            auc = roc_auc_score(y_true, scores)
        else:
            auc = np.nan
    except ValueError:
        auc = np.nan

    # 2. Optimal Threshold Search (Maximize F1 Score)
    best_f1 = 0.0
    # Search percentiles for the score (higher score = anomaly)
    threshold_candidates = np.linspace(np.percentile(scores, 90), np.max(scores), 50)
    best_threshold = np.percentile(scores, 95) # Default/initial guess based on contamination=0.05

    for threshold in threshold_candidates:
        # Prediction: scores > threshold means anomaly (1)
        y_pred = (scores > threshold).astype(int) 
        
        # Check if F1 can be computed
        if np.sum(y_pred) > 0: # Ensure positive predictions exist
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
    
    # Use the best threshold for final metrics
    y_pred_best = (scores > best_threshold).astype(int)
    
    # 3. Precision, Recall, F1, False Alarm Rate (FAR)
    # FAR is False Positive Rate (FPR) = FP / (FP + TN)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    precision = precision_score(y_true, y_pred_best, zero_division=0)
    recall = recall_score(y_true, y_pred_best, zero_division=0)
    f1 = best_f1
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "File": file_name,
        "Dataset": dataset_name,
        "Model": "LOF",
        "AUC": auc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "False Alarm Rate (FAR)": false_alarm_rate,
        "Optimal Threshold": best_threshold,
        "GT Anomaly Count": np.sum(y_true)
    }

# ======================
# Plot anomalies (Minor adjustment for correct index handling)
# ======================
def plot_corrected(df_original, df_corrected, anomalies, file_name):
    plt.figure(figsize=(14,6))
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_corrected.columns:
            plt.plot(df_original[col], label=f"{col} original", alpha=0.5)
            # Use the index of df_corrected for alignment
            plt.plot(df_corrected[col], label=f"{col} corrected", alpha=0.9)
    # Highlight the GT anomalies
    for anom in anomalies:
        if anom["vital"] in df_original.columns:
            # Assumes anom['index'] is the original DataFrame index (0-based)
            plt.scatter(anom["index"], df_original.loc[anom["index"], anom["vital"]], color="red", marker="x", s=50)
            
    plt.title(f"LOF Anomaly Correction - {file_name}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file.stem}_LOF_corrected_graph.png")
    plt.close()

# ======================
# Process all files - MODIFIED FOR EVALUATION
# ======================
evaluation_summary = []

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} (LOF) ===")
    try:
        df_original = pd.read_csv(file, low_memory=False)
        df = df_original.copy()
        
        # 1. Clean data and drop columns with no valid data
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        if df.empty:
            print("⚠️ No valid numeric data found.")
            continue
        
        # Ensure the DF has a simple 0-based index for GT alignment
        df = df.reset_index(drop=True)
        
        # 2. --- GROUND TRUTH GENERATION (BEFORE CLEANING) ---
        gt_array = np.zeros(len(df))
        anomalies_for_cleaning = label_anomalies(df, gt_array)
        y_true = gt_array

        # 3. Data Cleaning (to train the model on 'normal' data)
        df_clean = df.copy()
        # Replace GT anomalies with NaN
        for anom in anomalies_for_cleaning:
            df_clean.loc[anom['index'], anom['vital']] = np.nan

        # Impute, drop low variance columns
        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.loc[:, df_clean.nunique() > 1].fillna(df_clean.mean())

        if df_clean.empty or len(df_clean.columns) == 0:
             print("⚠️ Skipped: Data has no variance after cleaning.")
             continue
        
        # 4. Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean.values.astype(float))

        # 5. LOF Model
        # contamination=0.05 is the prior estimate of anomaly fraction
        lof = LocalOutlierFactor(n_neighbors=20, contamination='auto', novelty=False)
        
        # fit_predict runs fit and predicts the labels (-1 for anomaly, 1 for inlier)
        y_pred = lof.fit_predict(X_scaled) 
        
        # Negative Outlier Factor (score_samples returns the NOF where values close to -1 are inliers)
        # We need the anomaly score, where higher = more anomalous.
        # For LOF, the outlier score is -lof.negative_outlier_factor_
        anomaly_scores = -lof.negative_outlier_factor_

        # 6. Evaluation against Ground Truth
        dataset_name = file.stem
        eval_results = evaluate_anomaly_detection(y_true, anomaly_scores, file.name, dataset_name)
        evaluation_summary.append(eval_results)

        print(f"Metrics (Max F1 Threshold): AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | P={eval_results['Precision']:.4f} | R={eval_results['Recall']:.4f} | FAR={eval_results['False Alarm Rate (FAR)']:.4f}")
        print(f"GT Anomalies: {eval_results['GT Anomaly Count']}, LOF Detected: {np.sum(y_pred == -1)}")

        # 7. Data Correction (using the model's optimal prediction)
        # Use the optimal threshold found during evaluation for correction
        y_pred_best_mask = (anomaly_scores > eval_results["Optimal Threshold"])
        
        df_corrected = df_clean.copy() # Start from the cleaned data
        
        # Replace model-detected anomalies with NaN and impute/interpolate
        df_corrected.loc[y_pred_best_mask, :] = np.nan
        df_corrected = df_corrected.interpolate().ffill().bfill()
        
        # Save corrected CSV
        corrected_path = OUTPUT_DIR / f"{file.stem}_LOF_corrected.csv"
        df_corrected.to_csv(corrected_path, index=False)
        print(f"✅ LOF corrected data saved → {corrected_path}")

        # 8. Plot before/after
        plot_corrected(df, df_corrected, anomalies_for_cleaning, file.stem)
        
    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")

# ======================
# Save Overall Summary
# ======================
if evaluation_summary:
    eval_df = pd.DataFrame(evaluation_summary)
    eval_df = eval_df.sort_values(by="Dataset").reset_index(drop=True)
    
    summary_path = OUTPUT_DIR / "lof_performance_summary.csv"
    eval_df.to_csv(summary_path, index=False)
    print(f"\n✅ LOF Performance Summary saved → {summary_path}")
    print("\nSummary of Performance Metrics (LOF):")
    print(eval_df[["Dataset", "AUC", "F1", "Precision", "Recall", "False Alarm Rate (FAR)", "GT Anomaly Count"]])

else:
    print("\n⚠️ No results generated to summarize.")