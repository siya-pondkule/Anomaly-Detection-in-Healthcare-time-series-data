import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# ======================
# Parameters and Paths
# ======================
MODEL_NAME = "IsolationForest_Tuned"
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\CALLOUT.csv")
CONTAMINATION = 0.08 # Used only for thresholding
BASE_OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for CALLOUT")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"{MODEL_NAME}-CALLOUT-Results"
SUMMARY_DIR = BASE_OUTPUT_DIR / "Summary of Callout"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Data Preprocessing (Shared)
# ======================
def preprocess_callouts_ml(df):
    df['createtime'] = pd.to_datetime(df['createtime'])
    df['outcometime'] = pd.to_datetime(df['outcometime'])
    df['acknowledgetime'] = pd.to_datetime(df['acknowledgetime'])
    df['duration_hours'] = (df['outcometime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'] = (df['acknowledgetime'] - df['createtime']).dt.total_seconds() / 3600
    
    # FIX: Corrected fillna to remove FutureWarning
    df['ack_lag_hours'] = df['ack_lag_hours'].fillna(df['ack_lag_hours'].mean())
    
    features_df = df[[
        'duration_hours', 'ack_lag_hours',
        'request_tele', 'request_resp', 'request_cdiff', 'request_mrsa', 'request_vre',
        'curr_careunit', 'callout_service', 'acknowledge_status'
    ]].copy()
    features_df = pd.get_dummies(
        features_df, columns=['curr_careunit', 'callout_service', 'acknowledge_status'], drop_first=True
    )
    y_true = (df['callout_outcome'] == 'Cancelled').astype(int).values
    
    return features_df, y_true

# ======================
# Evaluation Function
# ======================
def evaluate_ml_model(y_true, anomaly_scores, contamination_rate):
    # Isolation Forest score: larger is more normal, so we flip the sign
    anomaly_scores = -anomaly_scores 
    
    try:
        auc = roc_auc_score(y_true, anomaly_scores)
    except ValueError:
        auc = np.nan
        
    # Thresholding based on contamination rate
    threshold = np.percentile(anomaly_scores, 100 * (1 - contamination_rate))
    y_pred = (anomaly_scores >= threshold).astype(int) 
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    results = {"Model": MODEL_NAME, "AUC": auc, "F1": f1, "Precision": precision, "Recall": recall, "FAR": far, "Threshold": threshold}
    return results

# ======================
# Main Processing (Semi-Supervised)
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on CALLOUT.csv (Trained on Normal Data) ===")
    
    df_raw = pd.read_csv(DATA_PATH)
    features_df, y_true = preprocess_callouts_ml(df_raw)

    # 1. Separate normal data for training
    X_train_normal = features_df[y_true == 0]
    X_full = features_df.values # Full dataset for testing

    # 2. Scale (fit only on normal data)
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train_normal)
    X_scaled_full = scaler.transform(X_full)
    
    # 3. Train Model only on scaled normal data
    # contamination is set low (0.01) to encourage the model to learn a tight boundary 
    # around the normal data, prioritizing a good fit over thresholding.
    model_if = IsolationForest(random_state=42, contamination=0.01) 
    model_if.fit(X_scaled_train)
    
    # 4. Predict anomaly scores on the full dataset
    anomaly_scores = model_if.decision_function(X_scaled_full)
    
    # 5. Evaluate on full dataset
    eval_results = evaluate_ml_model(y_true, anomaly_scores, CONTAMINATION)

    print(f"\nResults for CALLOUT.csv ({MODEL_NAME}): AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f}")

    # Save Summary
    summary_df = pd.DataFrame([eval_results])
    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_CALLOUT_Evaluation_Summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ {MODEL_NAME} Model execution complete.")