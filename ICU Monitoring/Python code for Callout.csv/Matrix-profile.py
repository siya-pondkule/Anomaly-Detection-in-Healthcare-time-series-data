import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
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
    
    y_true = (df['callout_outcome'] == 'Cancelled').astype(int).values
    
    return df['duration_hours'].values, y_true, df.index


# ======================
# Simplified Distance Profile
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
# Compute Metrics at Each Threshold
# ======================
def compute_metrics(y_true, anomaly_scores, threshold):
    y_pred = (anomaly_scores >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, anomaly_scores)
    except:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

    return {
        "Threshold_Value": threshold,
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FAR": fp / (fp + tn) if (fp + tn) > 0 else 0
    }


# ======================
# Main Processing
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on CALLOUT.csv (Window: {WINDOW_SIZE}) ===")

    df_raw = pd.read_csv(DATA_PATH)
    ts_duration, gt_array_full, original_indices = preprocess_callouts(df_raw)

    anomaly_scores = calculate_distance_profile(ts_duration, WINDOW_SIZE)
    gt_array = gt_array_full[WINDOW_SIZE - 1:]

    # ======================
    # Multi-Threshold Evaluation
    # ======================
    thresholds = {
        "90%": np.percentile(anomaly_scores, 90),
        "95%": np.percentile(anomaly_scores, 95),
        "98%": np.percentile(anomaly_scores, 98),
    }

    all_results = []
    best_result = None

    print("\n===== MULTI-THRESHOLD METRICS =====")
    for name, thr in thresholds.items():
        metrics = compute_metrics(gt_array, anomaly_scores, thr)
        metrics["Threshold_Name"] = name
        all_results.append(metrics)

        print(f"\n➡ Threshold {name} (value={thr:.6f})")
        print(f"AUC={metrics['AUC']:.4f} | Precision={metrics['Precision']:.4f} | "
              f"Recall={metrics['Recall']:.4f} | F1={metrics['F1']:.4f} | FAR={metrics['FAR']:.4f}")

        if best_result is None or metrics["F1"] > best_result["F1"]:
            best_result = metrics

    # ======================
    # Best Threshold Summary
    # ======================
    print("\n===== BEST THRESHOLD SELECTED =====")
    print(f"Best Threshold: {best_result['Threshold_Name']} (value={best_result['Threshold_Value']:.6f})")
    print(f"Best F1={best_result['F1']:.4f}, Precision={best_result['Precision']:.4f}, Recall={best_result['Recall']:.4f}")

    # ======================
    # SAVE SUMMARY
    # ======================
    summary_df = pd.DataFrame(all_results)
    summary_df["Model"] = MODEL_NAME
    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_CALLOUT_MultiThreshold_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")

    # ======================
    # PLOT
    # ======================
    plt.figure(figsize=(12,6))

    duration_plot_indices = original_indices[WINDOW_SIZE-1:]
    duration_data = ts_duration[WINDOW_SIZE-1:]

    plt.scatter(duration_plot_indices, duration_data, 
                c=anomaly_scores, cmap='viridis', s=20, label='Duration (Color by Profile Score)')

    best_thr = best_result["Threshold_Value"]
    y_pred = (anomaly_scores >= best_thr).astype(int)
    model_anomalies_indices = duration_plot_indices[y_pred == 1]

    plt.scatter(model_anomalies_indices, duration_data[y_pred == 1],
                color="red", marker="o", edgecolors='red', facecolors='none', s=100,
                label=f'Predicted Anomalies ({len(model_anomalies_indices)})')

    plt.title(f"CALLOUT Anomaly Detection ({MODEL_NAME}): Duration colored by Matrix Profile Score")
    plt.xlabel("Record Index")
    plt.ylabel("Callout Duration (Hours)")
    plt.colorbar(label='Matrix Profile Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "CALLOUT_anomaly_plot_duration.png")
    plt.close()

    print(f"✅ {MODEL_NAME} Model completed successfully. Results saved to → {OUTPUT_DIR}")
