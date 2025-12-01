import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# ======================
# Parameters and Paths
# ======================
MODEL_NAME = "LocalOutlierFactor"
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\CALLOUT.csv")
CONTAMINATION = 0.08
BASE_OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for CALLOUT")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"{MODEL_NAME}-CALLOUT-Results"
SUMMARY_DIR = BASE_OUTPUT_DIR / "Summary of Callout"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Data Preprocessing (Same as before)
# ======================
def preprocess_callouts_ml(df):
    df['createtime'] = pd.to_datetime(df['createtime'])
    df['outcometime'] = pd.to_datetime(df['outcometime'])
    df['acknowledgetime'] = pd.to_datetime(df['acknowledgetime'])
    
    df['duration_hours'] = (df['outcometime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'] = (df['acknowledgetime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'].fillna(df['ack_lag_hours'].mean(), inplace=True)
    
    features_df = df[[
        'duration_hours', 'ack_lag_hours',
        'request_tele', 'request_resp', 'request_cdiff', 'request_mrsa', 'request_vre',
        'curr_careunit', 'callout_service', 'acknowledge_status'
    ]].copy()

    features_df = pd.get_dummies(
        features_df, columns=['curr_careunit', 'callout_service', 'acknowledge_status'], drop_first=True
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    y_true = (df['callout_outcome'] == 'Cancelled').astype(int).values

    return X_scaled, y_true, features_df.columns, features_df['duration_hours'].values


# ======================
# Compute metrics for any threshold
# ======================
def compute_threshold_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, scores)
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
    print(f"\n=== Running {MODEL_NAME} on CALLOUT.csv ===")

    df_raw = pd.read_csv(DATA_PATH)
    X_scaled, y_true, feature_names, X_duration = preprocess_callouts_ml(df_raw)

    # LOF model (unchanged)
    model_lof = LocalOutlierFactor(n_neighbors=5, contamination=CONTAMINATION, novelty=True)
    model_lof.fit(X_scaled)

    raw_scores = model_lof.decision_function(X_scaled)

    # LOF anomaly score handling — flip so bigger = more anomalous
    anomaly_scores = -raw_scores

    # ============= MULTIPLE THRESHOLDS =============
    thresholds = {
        "90%": np.percentile(anomaly_scores, 90),
        "95%": np.percentile(anomaly_scores, 95),
        "98%": np.percentile(anomaly_scores, 98)
    }

    all_results = []
    best_result = None

    print("\n===== MULTI-THRESHOLD METRICS =====")

    for name, thr in thresholds.items():
        metrics = compute_threshold_metrics(y_true, anomaly_scores, thr)
        metrics["Threshold_Name"] = name
        all_results.append(metrics)

        print(f"\n➡ Threshold {name} (value={thr:.6f})")
        print(f"AUC={metrics['AUC']:.4f} | Precision={metrics['Precision']:.4f} | Recall={metrics['Recall']:.4f} "
              f"| F1={metrics['F1']:.4f} | FAR={metrics['FAR']:.4f}")

        # Best threshold = highest F1
        if best_result is None or metrics["F1"] > best_result["F1"]:
            best_result = metrics

    # ============= BEST THRESHOLD =============
    print("\n===== BEST THRESHOLD =====")
    print(f"Best Threshold: {best_result['Threshold_Name']} (value={best_result['Threshold_Value']:.6f})")
    print(f"Best F1={best_result['F1']:.4f}, Precision={best_result['Precision']:.4f}, Recall={best_result['Recall']:.4f}")

    # ============= SAVE SUMMARY CSV =============
    df_summary = pd.DataFrame(all_results)
    df_summary["Model"] = MODEL_NAME

    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_CALLOUT_MultiThreshold_Summary.csv"
    df_summary.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")
    print("\n✅ LOF Multi-Threshold Evaluation Completed.")
