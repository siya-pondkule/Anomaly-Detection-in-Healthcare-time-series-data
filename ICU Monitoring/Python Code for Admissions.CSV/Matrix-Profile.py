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
WINDOW_SIZE = 5
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
    y_true = df_filtered['hospital_expire_flag'].values

    return df_filtered['LOS_hours'].values, y_true, df_filtered.index


# ======================
# Simplified Matrix Profile Calculation
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
                subseq = ts_scaled[j:j + window_size]
                dist = np.sqrt(np.sum((query - subseq)**2))
                if dist < min_dist:
                    min_dist = dist

        profile[i] = min_dist

    return profile


# ======================
# Compute Metrics for a Threshold
# ======================
def compute_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, scores)
    except:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "Threshold": threshold,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far
    }


# ======================
# Main Processing
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on ADMISSIONS.csv ===")

    df_raw = pd.read_csv(DATA_PATH)
    ts_los, gt_array_full, original_indices = preprocess_admissions(df_raw)

    anomaly_scores = calculate_distance_profile(ts_los, WINDOW_SIZE)

    gt_array = gt_array_full[WINDOW_SIZE - 1:]


    # =====================================================
    # MULTI-THRESHOLD EVALUATION (90, 95, 98)
    # =====================================================
    threshold_values = {
        "90%": np.percentile(anomaly_scores, 90),
        "95%": np.percentile(anomaly_scores, 95),
        "98%": np.percentile(anomaly_scores, 98)
    }

    all_results = []
    best_result = None

    print("\n===== MULTI-THRESHOLD RESULTS =====")
    for name, th in threshold_values.items():
        metrics = compute_metrics(gt_array, anomaly_scores, th)
        metrics["Threshold_Name"] = name
        all_results.append(metrics)

        print(f"\nThreshold {name} (value={th:.6f})")
        print(f"AUC={metrics['AUC']:.4f} | Precision={metrics['Precision']:.4f} | "
              f"Recall={metrics['Recall']:.4f} | F1={metrics['F1']:.4f} | FAR={metrics['FAR']:.4f}")

        if best_result is None or metrics["F1"] > best_result["F1"]:
            best_result = metrics


    # =====================================================
    # Best Threshold
    # =====================================================
    print("\n===== BEST THRESHOLD =====")
    print(f"Best Threshold = {best_result['Threshold_Name']} ({best_result['Threshold']})")
    print(f"Best F1 = {best_result['F1']:.4f}")
    print(f"Precision={best_result['Precision']:.4f} | Recall={best_result['Recall']:.4f} | FAR={best_result['FAR']:.4f}")


    # =====================================================
    # Save Summary CSV
    # =====================================================
    summary_df = pd.DataFrame(all_results)
    summary_df["Model"] = MODEL_NAME
    summary_df["Reco_MSE"] = np.nan
    summary_df["Reco_R2"] = np.nan

    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_ADMISSIONS_MultiThreshold_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")


    # =====================================================
    # Plot Using Best Threshold
    # =====================================================
    best_thr = best_result["Threshold"]
    y_pred = (anomaly_scores >= best_thr).astype(int)

    los_plot_indices = original_indices[WINDOW_SIZE - 1:]
    los_data = ts_los[WINDOW_SIZE - 1:]

    anomaly_indices = los_plot_indices[y_pred == 1]

    plt.figure(figsize=(12, 6))
    plt.scatter(los_plot_indices, los_data, 
                c=anomaly_scores, cmap='viridis', s=20)

    plt.scatter(anomaly_indices,
                los_data[y_pred == 1],
                color='red', edgecolors='red', facecolors='none', s=100,
                label=f"Detected Anomalies ({len(anomaly_indices)})")

    plt.title(f"{MODEL_NAME} - LOS Anomalies Using Best Threshold ({best_result['Threshold_Name']})")
    plt.xlabel("Record Index")
    plt.ylabel("Length of Stay (Hours)")
    plt.colorbar(label='Matrix Profile Score')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ADMISSIONS_anomaly_plot_best_threshold.png")
    plt.close()

    print(f"\n{MODEL_NAME} Model completed successfully.")
