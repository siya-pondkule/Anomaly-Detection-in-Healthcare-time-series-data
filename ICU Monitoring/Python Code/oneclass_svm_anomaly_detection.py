import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------- Config ----------
DATASETS_DIR = Path(r"D:/Final Year/Project/Anomaly Detection/ICU Monitoring/Datasets")
OUTPUT_DIR = DATASETS_DIR / "../Results of all Models/OneClassSVM-ModelResult"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
# Detect and LABEL anomalies (Ground Truth) - MODIFIED
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
# Performance Metrics Calculation - NEW FUNCTION
# ======================
def evaluate_anomaly_detection(y_true, decision_scores, file_name, dataset_name):
    """
    Calculates all required performance metrics (AUC, F1, P, R, FAR) 
    using the OC-SVM decision scores.
    """
    
    # OC-SVM decision function: Positive for inliers, Negative for outliers.
    # Anomaly Score (scores where HIGHER = more anomalous) = -decision_scores
    anomaly_scores = -decision_scores 
    
    # 1. Check for valid ground truth
    if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        # Default FAR=0.05 is based on typical nu/contamination parameter
        return {
            "File": file_name, "Dataset": dataset_name, "Model": "OC-SVM",
            "AUC": np.nan, "F1": 0.0, "Precision": 0.0, "Recall": 0.0, 
            "False Alarm Rate (FAR)": 0.05, "Optimal Threshold": np.nan, "GT Anomaly Count": np.sum(y_true)
        }

    # 2. AUC 
    try:
        if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
            auc = roc_auc_score(y_true, anomaly_scores)
        else:
            auc = np.nan
    except ValueError:
        auc = np.nan

    # 3. Optimal Threshold Search (Maximize F1 Score)
    best_f1 = 0.0
    # Search threshold candidates across a wide range of the negative decision function
    # The anomaly threshold is usually close to 0 (where decision function is 0)
    # We search scores (which are -decision_scores)
    threshold_candidates = np.linspace(np.percentile(anomaly_scores, 10), np.max(anomaly_scores), 50)
    best_threshold = np.percentile(anomaly_scores, 95) # Initial guess 

    for threshold in threshold_candidates:
        # Prediction: scores > threshold means anomaly (1)
        y_pred = (anomaly_scores > threshold).astype(int) 
        
        # Check if F1 can be computed
        if np.sum(y_pred) > 0: # Ensure positive predictions exist
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
    
    # Use the best threshold for final metrics
    y_pred_best = (anomaly_scores > best_threshold).astype(int)
    
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
        "Model": "OC-SVM",
        "AUC": auc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "False Alarm Rate (FAR)": false_alarm_rate,
        "Optimal Threshold": best_threshold,
        "GT Anomaly Count": np.sum(y_true)
    }

# ======================
# Plot anomalies (Minor adjustment for clarity)
# ======================
def plot_corrected(df_original, df_corrected, anomalies, file_name, scores=None, threshold=None, score_label="Score"):
    plt.figure(figsize=(14,6))
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    
    # Plot data series on primary axis
    for col in numeric_cols:
        if col in df_corrected.columns:
            plt.plot(df_original[col], label=f"{col} original", alpha=0.5)
            plt.plot(df_corrected[col], label=f"{col} corrected", alpha=0.9)
    
    # Highlight GT anomalies
    for anom in anomalies:
        if anom["vital"] in df_original.columns:
            plt.scatter(anom["index"], df_original.loc[anom["index"], anom["vital"]], color="red", marker="x", s=50)

    # Plot scores on secondary axis
    if scores is not None:
        ax2 = plt.twinx()
        ax2.plot(scores, alpha=0.4, linestyle=':', color='gray', label=score_label)
        if threshold is not None:
            ax2.axhline(y=threshold, color='r', linestyle='--', label='Optimal Threshold')
        ax2.set_ylabel(score_label)
        ax2.legend(loc='upper right')

    plt.title(f"One-Class SVM Correction - {file_name}")
    plt.xlabel("Index")
    plt.ylabel("Vital Value")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file.stem}_ocsvm_correction.png")
    plt.close()


# ---------- Main Loop ----------
evaluation_summary = []

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} (One-Class SVM) ===")
    try:
        df_original = pd.read_csv(file, low_memory=False)
        df = df_original.copy()
        
        # 1. Data Cleaning and Index Reset
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').reset_index(drop=True)
        if df.empty:
            print("No numeric cols.")
            continue

        # 2. --- GROUND TRUTH GENERATION (BEFORE CLEANING) ---
        gt_array = np.zeros(len(df))
        threshold_anoms = label_anomalies(df, gt_array)
        y_true = gt_array

        # 3. Data Cleaning (to train on 'normal' data)
        df_clean = df.copy()
        for a in threshold_anoms:
            if a['vital'] in df_clean.columns:
                df_clean.loc[a['index'], a['vital']] = np.nan
        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.loc[:, df_clean.nunique() > 1].fillna(df_clean.mean())
        
        if df_clean.empty or len(df_clean.columns) == 0:
             print("⚠️ Skipped: Data has no variance after cleaning.")
             continue

        # 4. Scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean.values.astype(float))

        # 5. One-Class SVM Model
        oc = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale') # nu is the expected anomaly fraction
        oc.fit(X)
        
        # Decision Score: Positive for inliers, Negative for outliers.
        decision_scores = oc.decision_function(X) 
        
        # Anomaly Score (Higher = more anomalous)
        anomaly_scores = -decision_scores

        # 6. Evaluation against Ground Truth
        dataset_name = file.stem
        eval_results = evaluate_anomaly_detection(y_true, decision_scores, file.name, dataset_name)
        evaluation_summary.append(eval_results)

        print(f"Metrics (Max F1 Threshold): AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | P={eval_results['Precision']:.4f} | R={eval_results['Recall']:.4f} | FAR={eval_results['False Alarm Rate (FAR)']:.4f}")
        print(f"GT Anomalies: {eval_results['GT Anomaly Count']}")

        # 7. Data Correction (using the model's optimal prediction)
        # Use the optimal threshold found during evaluation (scores > threshold is anomaly)
        y_pred_best_mask = (anomaly_scores > eval_results["Optimal Threshold"])
        
        df_corrected = df_clean.copy() 
        df_corrected.loc[y_pred_best_mask, :] = np.nan
        df_corrected = df_corrected.interpolate().ffill().bfill()

        # 8. Save corrected CSV and Plot
        corrected_path = OUTPUT_DIR / f"{file.stem}_corrected_oneclasssvm.csv"
        df_corrected.to_csv(corrected_path, index=False)
        
        plot_corrected(df, df_corrected, threshold_anoms, file.stem, 
                       scores=anomaly_scores, threshold=eval_results["Optimal Threshold"], 
                       score_label="Anomaly Score (-Decision Function)")

    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")

# ---------- Overall Accuracy Summary ----------
if evaluation_summary:
    eval_df = pd.DataFrame(evaluation_summary)
    eval_df = eval_df.sort_values(by="Dataset").reset_index(drop=True)
    
    summary_path = OUTPUT_DIR / "oneclasssvm_performance_summary.csv"
    eval_df.to_csv(summary_path, index=False)
    print(f"\n✅ One-Class SVM Performance Summary saved → {summary_path}")
    print("\nSummary of Performance Metrics (One-Class SVM):")
    print(eval_df[["Dataset", "AUC", "F1", "Precision", "Recall", "False Alarm Rate (FAR)", "GT Anomaly Count"]])

else:
    print("⚠️ No results to summarize.")