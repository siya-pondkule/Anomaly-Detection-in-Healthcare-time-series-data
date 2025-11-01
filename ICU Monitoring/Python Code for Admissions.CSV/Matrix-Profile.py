import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ======================
# Parameters and Paths
# ======================
MODEL_NAME = "Matrix_Profile_LOS"
WINDOW_SIZE = 5 # Subsequence length for profile calculation
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ADMISSIONS.csv")
BASE_OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ADMISSION")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"{MODEL_NAME}-ADMISSIONS-Results"
SUMMARY_DIR = BASE_OUTPUT_DIR / "Summary of Admission"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Data Preprocessing
# ======================
def preprocess_admissions(df):
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df['LOS_hours'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / 3600
    df_filtered = df.dropna(subset=['LOS_hours']).copy()
    
    # Ground Truth: Hospital Expire Flag (1 = anomaly/adverse outcome)
    y_true = df_filtered['hospital_expire_flag'].values
    
    return df_filtered['LOS_hours'].values, y_true, df_filtered.index


# ======================
# Simplified Distance Profile (Manual NumPy Implementation)
# ======================
def calculate_distance_profile(ts, window_size):
    n = len(ts)
    profile_len = n - window_size + 1
    profile = np.full(profile_len, np.inf)

    # Normalize the time series for robust distance calculation (important for time series)
    ts_scaled = (ts - np.mean(ts)) / np.std(ts)
    
    for i in range(profile_len):
        query = ts_scaled[i:i + window_size]
        min_dist = np.inf
        for j in range(profile_len):
            if abs(i - j) > window_size // 2: # Exclude trivial match
                subsequence = ts_scaled[j:j + window_size]
                # Euclidean Distance
                dist = np.sqrt(np.sum((query - subsequence)**2))
                if dist < min_dist:
                    min_dist = dist
        profile[i] = min_dist
    
    return profile


# ======================
# Evaluation Function (Finds best threshold for F1-score)
# (Adjusted to account for the profile's shorter length)
# ======================
def evaluate_anomaly_detection(y_true, anomaly_scores):
    try:
        auc = roc_auc_score(y_true, anomaly_scores)
    except ValueError:
        auc = np.nan

    best_f1, best_threshold = 0, np.percentile(anomaly_scores, 95) # Higher default percentile for MP/time-series
    
    if np.sum(y_true) == 0:
        return {"AUC": auc, "F1": np.nan, "Precision": np.nan, "Recall": np.nan, "FAR": np.nan, "Threshold": best_threshold}

    for q in np.linspace(90, 99.9, 100):
        threshold = np.percentile(anomaly_scores, q)
        y_pred = (anomaly_scores >= threshold).astype(int)
        
        if np.sum(y_pred) > 0:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
    
    y_pred_best = (anomaly_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "AUC": auc, "F1": best_f1,
        "Precision": precision_score(y_true, y_pred_best, zero_division=0),
        "Recall": recall_score(y_true, y_pred_best, zero_division=0),
        "FAR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "Threshold": best_threshold
    }


# ======================
# Main Processing
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on ADMISSIONS.csv (Window: {WINDOW_SIZE}) ===")
    
    df_raw = pd.read_csv(DATA_PATH)
    ts_los, gt_array_full, original_indices = preprocess_admissions(df_raw)
    
    # Calculate the simplified Distance Profile
    # High score means it's far from its nearest neighbor (i.e., highly anomalous)
    anomaly_scores = calculate_distance_profile(ts_los, WINDOW_SIZE)
    
    # The anomaly score corresponds to indices from WINDOW_SIZE - 1 onwards
    gt_array = gt_array_full[WINDOW_SIZE - 1:]

    # Evaluate
    eval_results = evaluate_anomaly_detection(gt_array, anomaly_scores)

    print(f"\nResults for ADMISSIONS.csv ({MODEL_NAME}):")
    print(f"AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f}")

    # Save Summary
    summary_data = {"Model": [MODEL_NAME], "AUC": [eval_results['AUC']], "F1": [eval_results['F1']], "Precision": [eval_results['Precision']], "Recall": [eval_results['Recall']], "FAR": [eval_results['FAR']], "Reco_MSE": [np.nan], "Reco_R2": [np.nan]}
    summary_df = pd.DataFrame(summary_data)
    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_ADMISSIONS_Evaluation_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    # Plot
    plt.figure(figsize=(12,6))
    
    # Plot the LOS time series, colored by the Matrix Profile score
    los_plot_indices = original_indices[WINDOW_SIZE-1:]
    los_data = ts_los[WINDOW_SIZE-1:]
    
    plt.scatter(los_plot_indices, los_data, 
                c=anomaly_scores, cmap='viridis', s=20, label='LOS (Color by Profile Score)')
    
    # Identify model-detected anomalies based on best threshold
    y_pred_model = (anomaly_scores >= eval_results['Threshold']).astype(int)
    model_anomalies_indices = los_plot_indices[y_pred_model == 1]
    
    # Mark the predicted anomalies
    plt.scatter(model_anomalies_indices, los_data[y_pred_model == 1], 
                color="red", marker="o", edgecolors='red', facecolors='none', s=100, 
                label=f'Predicted Anomalies ({len(model_anomalies_indices)})')

    plt.title(f"ADMISSIONS Anomaly Detection ({MODEL_NAME}): LOS colored by Matrix Profile Score (Window: {WINDOW_SIZE})")
    plt.xlabel("Record Index")
    plt.ylabel("Length of Stay (Hours)")
    plt.colorbar(label='Simplified Matrix Profile Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ADMISSIONS_anomaly_plot_LOS.png")
    plt.close()

    print(f"✅ {MODEL_NAME} Model completed successfully. Results saved to → {OUTPUT_DIR}")