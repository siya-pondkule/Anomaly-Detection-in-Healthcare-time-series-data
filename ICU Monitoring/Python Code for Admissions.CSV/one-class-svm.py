import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ======================
# Paths and Parameters
# ======================
MODEL_NAME = "OneClassSVM"
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
    
    features_df = df[['LOS_hours','admission_type','discharge_location','ethnicity']].copy()
    features_df = pd.get_dummies(
        features_df, 
        columns=['admission_type','discharge_location','ethnicity'], 
        drop_first=True
    )
    features_df['LOS_hours'].fillna(features_df['LOS_hours'].mean(), inplace=True)
    
    y_true = df['hospital_expire_flag'].values
    return features_df, y_true


# ======================
# Helper: Compute Metrics for Threshold
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
        "Threshold_Value": threshold,
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
    X_processed, gt_array = preprocess_admissions(df_raw)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    # Train OCSVM
    ocsvm_model = OneClassSVM(nu=0.05, kernel="rbf", gamma='auto')
    ocsvm_model.fit(X_scaled)

    # Decision function gives higher = NORMAL → invert it
    raw_scores = ocsvm_model.decision_function(X_scaled)
    anomaly_scores = -raw_scores.flatten()

    print("Training complete.")

    # =====================================================
    # MULTI-THRESHOLD EVALUATION
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

        print(f"\nThreshold {name} (Value={th:.6f})")
        print(f"AUC={metrics['AUC']:.4f} | Precision={metrics['Precision']:.4f} | "
              f"Recall={metrics['Recall']:.4f} | F1={metrics['F1']:.4f} | FAR={metrics['FAR']:.4f}")

        if best_result is None or metrics["F1"] > best_result["F1"]:
            best_result = metrics


    # =====================================================
    # BEST THRESHOLD
    # =====================================================
    print("\n===== BEST THRESHOLD FOUND =====")
    print(f"Best Threshold = {best_result['Threshold_Name']}  (Value={best_result['Threshold_Value']})")
    print(f"Best F1 Score = {best_result['F1']:.4f}")
    print(f"Precision={best_result['Precision']:.4f} | Recall={best_result['Recall']:.4f} | FAR={best_result['FAR']:.4f}")


    # =====================================================
    # SAVE MULTI-THRESHOLD SUMMARY
    # =====================================================
    summary_df = pd.DataFrame(all_results)
    summary_df["Model"] = MODEL_NAME
    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_ADMISSIONS_MultiThreshold_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")


    # =====================================================
    # PLOT USING BEST THRESHOLD
    # =====================================================
    best_thr = best_result["Threshold_Value"]
    y_pred_best = (anomaly_scores >= best_thr).astype(int)
    anomaly_indices = X_processed.index[y_pred_best == 1]

    plt.figure(figsize=(12,6))
    plt.scatter(X_processed.index, X_processed['LOS_hours'], 
                c=anomaly_scores, cmap='viridis', s=20)

    plt.scatter(anomaly_indices,
                X_processed.loc[anomaly_indices,'LOS_hours'],
                color="red", edgecolors="red", facecolors="none", s=100,
                label=f"Detected Anomalies ({len(anomaly_indices)})")

    plt.title(f"{MODEL_NAME} - LOS Anomalies Using Best Threshold ({best_result['Threshold_Name']})")
    plt.xlabel("Record Index")
    plt.ylabel("Length of Stay (Hours)")
    plt.colorbar(label="Anomaly Score (Inverted OCSVM)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ADMISSIONS_best_threshold_plot.png")
    plt.close()

    print("\nOne-Class SVM Multi-Threshold Processing Completed Successfully.")
