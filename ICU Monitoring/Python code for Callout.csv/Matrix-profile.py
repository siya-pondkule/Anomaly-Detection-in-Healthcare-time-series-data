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
MODEL_NAME = "Matrix_Profile_Duration"
WINDOW_SIZE = 5 
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\CALLOUT.csv")
BASE_OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for CALLOUT")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"{MODEL_NAME}-CALLOUT-Results"
SUMMARY_DIR = BASE_OUTPUT_DIR / "Summary of Callout"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Data Preprocessing
# ======================
def preprocess_callouts(df):
    df['createtime'] = pd.to_datetime(df['createtime'])
    df['outcometime'] = pd.to_datetime(df['outcometime'])
    df['duration_hours'] = (df['outcometime'] - df['createtime']).dt.total_seconds() / 3600
    
    # Ground Truth: 'Cancelled' outcome is the anomaly (1)
    y_true = (df['callout_outcome'] == 'Cancelled').astype(int).values
    
    return df['duration_hours'].values, y_true, df.index


# ======================
# Simplified Distance Profile (Manual NumPy Implementation)
# ======================
def calculate_distance_profile(ts, window_size):
    n = len(ts)
    profile_len = n - window_size + 1
    profile = np.full(profile_len, np.inf)

    ts_scaled = (ts - np.mean(ts)) / np.std(ts)
    
    for i in range(profile_len):
        query = ts_scaled[i:i + window_size]
        min_dist = np.inf
        for j in range(profile_len):
            if abs(i - j) > window_size // 2: 
                subsequence = ts_scaled[j:j + window_size]
                dist = np.sqrt(np.sum((query - subsequence)**2))
                if dist < min_dist:
                    min_dist = dist
        profile[i] = min_dist
    
    return profile


# ======================
# Evaluation Function
# ======================
def evaluate_anomaly_detection(y_true, anomaly_scores):
    try:
        auc = roc_auc_score(y_true, anomaly_scores)
    except ValueError:
        auc = np.nan

    best_f1, best_threshold = 0, np.percentile(anomaly_scores, 95) 
    
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
    print(f"\n=== Running {MODEL_NAME} on CALLOUT.csv (Window: {WINDOW_SIZE}) ===")
    
    df_raw = pd.read_csv(DATA_PATH)
    ts_duration, gt_array_full, original_indices = preprocess_callouts(df_raw)
    
    # Calculate the simplified Distance Profile
    anomaly_scores = calculate_distance_profile(ts_duration, WINDOW_SIZE)
    
    # Adjust ground truth length
    gt_array = gt_array_full[WINDOW_SIZE - 1:]

    # Evaluate
    eval_results = evaluate_anomaly_detection(gt_array, anomaly_scores)

    print(f"\nResults for CALLOUT.csv ({MODEL_NAME}):")
    print(f"Ground Truth ('Cancelled' outcome): {np.sum(gt_array)} records out of {len(gt_array)}")
    print(f"AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f}")

    # Save Summary
    summary_data = {"Model": [MODEL_NAME], "AUC": [eval_results['AUC']], "F1": [eval_results['F1']], "Precision": [eval_results['Precision']], "Recall": [eval_results['Recall']], "FAR": [eval_results['FAR']], "Reco_MSE": [np.nan], "Reco_R2": [np.nan]}
    summary_df = pd.DataFrame(summary_data)
    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_CALLOUT_Evaluation_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    # Plot
    plt.figure(figsize=(12,6))
    
    duration_plot_indices = original_indices[WINDOW_SIZE-1:]
    duration_data = ts_duration[WINDOW_SIZE-1:]
    
    plt.scatter(duration_plot_indices, duration_data, 
                c=anomaly_scores, cmap='viridis', s=20, label='Duration (Color by Profile Score)')
    
    y_pred_model = (anomaly_scores >= eval_results['Threshold']).astype(int)
    model_anomalies_indices = duration_plot_indices[y_pred_model == 1]
    
    plt.scatter(model_anomalies_indices, duration_data[y_pred_model == 1], 
                color="red", marker="o", edgecolors='red', facecolors='none', s=100, 
                label=f'Predicted Anomalies ({len(model_anomalies_indices)})')

    plt.title(f"CALLOUT Anomaly Detection ({MODEL_NAME}): Callout Duration colored by Matrix Profile Score (Window: {WINDOW_SIZE})")
    plt.xlabel("Record Index")
    plt.ylabel("Callout Duration (Hours)")
    plt.colorbar(label='Simplified Matrix Profile Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "CALLOUT_anomaly_plot_duration.png")
    plt.close()

    print(f"✅ {MODEL_NAME} Model completed successfully. Results saved to → {OUTPUT_DIR}")