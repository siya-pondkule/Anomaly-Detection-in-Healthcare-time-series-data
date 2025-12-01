import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# ======================
# Parameters and Paths
# ======================
MODEL_NAME = "OneClassSVM"
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\CALLOUT.csv")
CONTAMINATION = 0.08 
BASE_OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for CALLOUT")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"{MODEL_NAME}-CALLOUT-Results"
SUMMARY_DIR = BASE_OUTPUT_DIR / "Summary of Callout"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Data Preprocessing 
# ======================
def preprocess_callouts_ml(df):
    df['createtime'] = pd.to_datetime(df['createtime'])
    df['outcometime'] = pd.to_datetime(df['outcometime'])
    df['acknowledgetime'] = pd.to_datetime(df['acknowledgetime'])
    
    df['duration_hours'] = (df['outcometime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'] = (df['acknowledgetime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'] = df['ack_lag_hours'].fillna(df['ack_lag_hours'].mean())   
     
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
# Threshold-based Evaluation Function
# ======================
def evaluate_at_threshold(y_true, anomaly_scores, threshold):
    # OCSVM: lower score = more anomalous
    y_pred = (anomaly_scores <= threshold).astype(int)
    
    try:
        auc = roc_auc_score(y_true, anomaly_scores)
    except:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

    return {
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FAR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "Threshold_Value": threshold
    }

# ======================
# Main Processing
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on CALLOUT.csv ===")
    
    df_raw = pd.read_csv(DATA_PATH)
    X_scaled, y_true, feature_names, X_duration = preprocess_callouts_ml(df_raw)
    
    # Train One-Class SVM
    model_ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=CONTAMINATION) 
    model_ocsvm.fit(X_scaled)
    anomaly_scores = model_ocsvm.decision_function(X_scaled)

    # ======================
    # Multi-Threshold Evaluation
    # ======================
    thresholds = {
        "90%": np.percentile(anomaly_scores, 10),   # lowest scores = anomalies
        "95%": np.percentile(anomaly_scores, 5),
        "98%": np.percentile(anomaly_scores, 2),
    }

    all_results = []
    best_result = None

    print("\n===== MULTI-THRESHOLD RESULTS =====")
    for name, thr in thresholds.items():
        metrics = evaluate_at_threshold(y_true, anomaly_scores, thr)
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
    # Save Summary CSV
    # ======================
    summary_df = pd.DataFrame(all_results)
    summary_df["Model"] = MODEL_NAME

    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_CALLOUT_MultiThreshold_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")
    print(f"✅ {MODEL_NAME} Model execution complete.")
