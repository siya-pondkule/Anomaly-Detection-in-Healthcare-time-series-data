import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
# Suppress convergence warnings often seen in ML models
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# ======================
# Directories
# ======================
# NOTE: Using a relative path structure, assume 'Datasets' is at a known location
DATASETS_DIR = Path(r"D:/Final Year/Project/Anomaly Detection/ICU Monitoring/Datasets")
OUTPUT_DIR = DATASETS_DIR / "../Results of all Models/Isolation-ModelResults"
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
def label_anomalies(series, col_name, gt_array, index_offset):
    """Detects physiological anomalies and updates the Ground Truth array."""
    # Ensure series is cleaned of non-numeric data for thresholding
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if series.empty:
        return []

    vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
    anomalous_indices = []

    # --- ANOMALY DETECTION LOGIC (based on TH) ---
    if vital:
        if vital == "HR":
            anomalous_indices += list(series[series > TH["HR_tachy"]].index)
            anomalous_indices += list(series[series < TH["HR_brady"]].index)
        elif vital == "SBP":
            anomalous_indices += list(series[series > TH["SBP_high"]].index)
            anomalous_indices += list(series[series < TH["SBP_low"]].index)
        elif vital == "DBP":
            anomalous_indices += list(series[series > TH["DBP_high"]].index)
            anomalous_indices += list(series[series < TH["DBP_low"]].index)
        elif vital == "MAP":
            anomalous_indices += list(series[series > TH["MAP_high"]].index)
            anomalous_indices += list(series[series < TH["MAP_low"]].index)
        elif vital == "RR":
            anomalous_indices += list(series[series > TH["RR_tachypnea"]].index)
            anomalous_indices += list(series[series < TH["RR_apnea"]].index)
        elif vital == "SpO2":
            anomalous_indices += list(series[series < TH["SpO2_low"]].index)
        elif vital == "Temp":
            anomalous_indices += list(series[series > TH["Temp_high"]].index)
            anomalous_indices += list(series[series < TH["Temp_low"]].index)
    else:
        # generic z-score detection
        z = (series - series.mean()) / series.std(ddof=0)
        anomalous_indices += list(series[np.abs(z) > 3].index)

    # Update the Ground Truth array (1 for anomaly)
    # The index_offset is 0 here because we are labeling the original DF before cleaning.
    for i in anomalous_indices:
        if i < len(gt_array):
            gt_array[i] = 1
            
    # Return the list of anomaly dictionaries for cleaning/plotting
    anomalies_list = [{"vital": col_name, "index": i} for i in anomalous_indices]
    return anomalies_list



# ======================
# Performance Metrics Calculation
# ======================
def evaluate_anomaly_detection(y_true, anomaly_scores, file_name, dataset_name):
    """
    Calculates all required performance metrics (AUC, F1, P, R, FAR) 
    using the anomaly scores (higher = more anomalous).
    """
    
    # Isolation Forest scores (decision_function) are NEGATIVE for anomalies.
    # We must invert the scores for AUC calculation where higher scores mean positive class (anomaly=1).
    scores = -anomaly_scores 
    
    # 1. AUC 
    try:
        # Need at least one positive (1) and one negative (0) class for AUC
        if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
            auc = roc_auc_score(y_true, scores)
        else:
            auc = np.nan
    except ValueError:
        auc = np.nan

    # 2. Optimal Threshold Search (Maximize F1 Score)
    best_f1 = 0
    # Search percentiles for the Isolation Forest score (negative value) 
    # The anomaly is on the *lower* end of the decision_function
    # Let's search across the scores themselves
    threshold_candidates = np.linspace(np.min(anomaly_scores), np.percentile(anomaly_scores, 10), 50)
    best_threshold = np.percentile(anomaly_scores, 5) # Default/initial guess

    for threshold in threshold_candidates:
        # Prediction: -1 is anomaly in Isolation Forest. Here, anomaly_scores < threshold means anomaly.
        y_pred = (anomaly_scores < threshold).astype(int) 
        
        # Check if F1 can be computed
        if np.sum(y_pred) > 0 and np.sum(y_true) > 0:
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
    
    # Use the best threshold for final metrics
    y_pred_best = (anomaly_scores < best_threshold).astype(int)
    
    # 3. Precision, Recall, F1, False Alarm Rate (FAR)
    # FAR is defined as False Positive Rate (FPR) = FP / (FP + TN)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0) # Handle edge cases (no anomalies/no predictions)
    
    precision = precision_score(y_true, y_pred_best, zero_division=0)
    recall = recall_score(y_true, y_pred_best, zero_division=0)
    f1 = best_f1
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "File": file_name,
        "Dataset": dataset_name,
        "Model": "Isolation Forest",
        "AUC": auc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "False Alarm Rate (FAR)": false_alarm_rate,
        "Optimal Threshold": best_threshold,
        "GT Anomaly Count": np.sum(y_true)
    }

# ======================
# Plot corrected vs original (Remains the same)
# ======================
def plot_corrected(df_original, df_corrected, anomalies, file_name):
    plt.figure(figsize=(14,6))
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_corrected.columns:
            # We must ensure the indices align for plotting, reset_index helps
            plot_df_orig = df_original[col].reset_index(drop=True)
            plot_df_corr = df_corrected[col].reset_index(drop=True)
            plt.plot(plot_df_orig, label=f"{col} (original)", alpha=0.5)
            plt.plot(plot_df_corr, label=f"{col} (corrected)", alpha=0.9)
    
    # Scatter plot for GT anomalies (Indices correspond to the original, un-cleaned data frame)
    for anom in anomalies:
        if anom["vital"] in df_original.columns:
            plt.scatter(anom["index"], df_original.loc[anom["index"], anom["vital"]], color="red", marker="x", s=50)

    plt.title(f"Anomaly Correction (Isolation Forest) - {file_name}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file.stem}_isoforest_plot.png")
    plt.close()


# ======================
# Main processing loop - MODIFIED FOR EVALUATION
# ======================
evaluation_summary = []
reconstruction_summary = [] # For the MSE/R2 between cleaned and corrected

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} (Isolation Forest) ===")
    try:
        df_original = pd.read_csv(file, low_memory=False)
        df = df_original.copy()
        
        # 1. Clean data and drop columns with no valid data
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        if df.empty:
            print("⚠️ Skipped empty dataset.")
            continue
        
        # 2. --- GROUND TRUTH GENERATION (BEFORE CLEANING) ---
        gt_array = np.zeros(len(df))
        anomalies_for_cleaning = []
        for col in df.columns:
            # The label_anomalies function modifies gt_array in place
            anomalies_for_cleaning.extend(label_anomalies(df[col], col, gt_array, index_offset=0))
        
        # Filter GT array to only include rows present in df (should be all)
        # GT_array now serves as Y_true for evaluation
        y_true = gt_array

        # 3. Data Cleaning (to train the model on 'normal' data)
        df_clean = df.copy()
        # Replace GT anomalies with NaN and impute/interpolate
        for a in anomalies_for_cleaning:
            if a['vital'] in df_clean.columns:
                df_clean.loc[a['index'], a['vital']] = np.nan
        df_clean = df_clean.interpolate().ffill().bfill()
        
        # Ensure only columns with variance are kept (and align with scaler)
        df_clean = df_clean.loc[:, df_clean.nunique() > 1].fillna(df_clean.mean())
        if df_clean.empty:
             print("⚠️ Skipped: Data has no variance after cleaning.")
             continue
        
        # 4. Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)

        # 5. Isolation Forest Model
        # contamination: Expected proportion of outliers in the dataset (a hyperparameter)
        model = IsolationForest(contamination='auto', random_state=42) # Use 'auto' or a fixed value like 0.05
        model.fit(X_scaled)

        # Anomaly Score (decision_function: lower = more anomalous)
        anomaly_scores = model.decision_function(X_scaled)

        # 6. Evaluation against Ground Truth
        dataset_name = file.stem
        eval_results = evaluate_anomaly_detection(y_true, anomaly_scores, file.name, dataset_name)
        evaluation_summary.append(eval_results)

        print(f"Metrics (Max F1 Threshold): AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | P={eval_results['Precision']:.4f} | R={eval_results['Recall']:.4f} | FAR={eval_results['False Alarm Rate (FAR)']:.4f}")

        # 7. Data Correction (using the model's prediction)
        # We use the optimal threshold found during evaluation for correction
        y_pred_best = (anomaly_scores < eval_results["Optimal Threshold"])
        
        df_corrected = df.copy() # Start from the original data
        # Only keep columns that were scaled
        df_corrected = df_corrected.loc[:, df_clean.columns] 
        
        # Replace model-detected anomalies with NaN
        df_corrected.loc[y_pred_best, :] = np.nan
        df_corrected = df_corrected.interpolate().ffill().bfill()
        
        # Save corrected data
        out_path = OUTPUT_DIR / f"{file.stem}_corrected_isoforest.csv"
        df_corrected.to_csv(out_path, index=False)
        print(f"✅ Corrected data saved → {out_path}")

        # 8. Plot graph
        plot_corrected(df, df_corrected, anomalies_for_cleaning, file.stem)
        
        # 9. Reconstruction/Correction Metrics (optional but useful)
        # Compare the cleaned data (GT anomalies imputed) vs. the corrected data (IF anomalies imputed)
        mse = mean_squared_error(df_clean.values, df_corrected.values)
        r2 = r2_score(df_clean.values, df_corrected.values)
        # Accuracy is calculated as 1 - normalized MSE
        acc = 100 * (1 - mse / np.var(df_clean.values))

        print(f"Correction MSE={mse:.6f} | R²={r2:.4f} | Correction Accuracy≈{acc:.2f}% | GT Anomalies={eval_results['GT Anomaly Count']}")

        reconstruction_summary.append({
            "File": file.name,
            "MSE": mse,
            "R2": r2,
            "Accuracy (%)": acc,
            "GT Anomaly Count": eval_results["GT Anomaly Count"]
        })

    except Exception as e:
        print(f"❌ Error in {file.name}: {e}")

# ======================
# Save Overall Summaries
# ======================
if evaluation_summary:
    # 1. Performance Metrics Summary (AUC, F1, P, R, FAR)
    eval_df = pd.DataFrame(evaluation_summary)
    eval_summary_path = OUTPUT_DIR / "isolationforest_performance_summary.csv"
    eval_df.to_csv(eval_summary_path, index=False)
    print(f"\n✅ Isolation Forest Performance Summary saved → {eval_summary_path}")
    print("\nSummary of Performance Metrics (Isolation Forest):")
    print(eval_df[["Dataset", "AUC", "F1", "Precision", "Recall", "False Alarm Rate (FAR)", "GT Anomaly Count"]])

    # 2. Reconstruction/Correction Summary (MSE, R2)
    reco_df = pd.DataFrame(reconstruction_summary)
    reco_summary_path = OUTPUT_DIR / "isolationforest_reconstruction_summary.csv"
    reco_df.to_csv(reco_summary_path, index=False)
    print(f"\n✅ Isolation Forest Reconstruction Summary saved → {reco_summary_path}")

else:
    print("\n⚠️ No results generated to summarize.")