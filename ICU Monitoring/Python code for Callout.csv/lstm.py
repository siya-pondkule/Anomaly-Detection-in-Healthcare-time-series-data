import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import tensorflow as tf

# Set seed for reproducibility
tf.keras.utils.set_random_seed(42)

# ======================
# Parameters and Paths
# ======================
MODEL_NAME = "LSTM_Autoencoder"
SEQUENCE_LENGTH = 5 
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\CALLOUT.csv")
BASE_OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for CALLOUT")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"{MODEL_NAME}-CALLOUT-Results"
SUMMARY_DIR = BASE_OUTPUT_DIR / "Summary of Callout"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Preprocessing
# ======================
def preprocess_callouts(df):
    df['createtime'] = pd.to_datetime(df['createtime'])
    df['outcometime'] = pd.to_datetime(df['outcometime'])
    df['acknowledgetime'] = pd.to_datetime(df['acknowledgetime'])
    
    df['duration_hours'] = (df['outcometime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'] = (df['acknowledgetime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'].fillna(df['ack_lag_hours'].mean(), inplace=True)
    
    features_df = df[[
        'duration_hours', 'ack_lag_hours',
        'request_tele', 'request_resp', 'request_cdiff',
        'request_mrsa', 'request_vre',
        'curr_careunit', 'callout_service', 'acknowledge_status'
    ]].copy()
    
    features_df = pd.get_dummies(
        features_df, 
        columns=['curr_careunit', 'callout_service', 'acknowledge_status'], 
        drop_first=True
    )
    
    y_true = (df['callout_outcome'] == 'Cancelled').astype(int).values
    return features_df, y_true

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X)

# ======================
# Model Definition
# ======================
def build_lstm_autoencoder(seq_len, n_features):
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.LSTM(32, activation='relu', return_sequences=False),
        layers.RepeatVector(seq_len),
        layers.LSTM(32, activation='relu', return_sequences=True),
        layers.TimeDistributed(layers.Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ======================
# Compute Metrics for a Given Threshold
# ======================
def compute_metrics(y_true, mse_scores, threshold):
    y_pred = (mse_scores >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, mse_scores)
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
# Main Execution
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on CALLOUT.csv ===")
    
    df_raw = pd.read_csv(DATA_PATH)
    X_processed, gt_array_full = preprocess_callouts(df_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    n_features = X_scaled.shape[1]

    X_seq = create_sequences(X_scaled, SEQUENCE_LENGTH)
    gt_array = gt_array_full[SEQUENCE_LENGTH - 1:]

    lstm_autoencoder = build_lstm_autoencoder(SEQUENCE_LENGTH, n_features)
    lstm_autoencoder.fit(X_seq, X_seq, epochs=50, batch_size=4, validation_split=0.1, verbose=0)
    
    reconstructions = lstm_autoencoder.predict(X_seq, verbose=0)
    mse_per_sequence = np.mean(np.mean(np.square(X_seq - reconstructions), axis=2), axis=1)

    # ====================== MULTI-THRESHOLD LOGIC ======================
    thresholds = {
        "90%": np.percentile(mse_per_sequence, 90),
        "95%": np.percentile(mse_per_sequence, 95),
        "98%": np.percentile(mse_per_sequence, 98)
    }

    all_results = []
    best_result = None

    print("\n===== MULTI-THRESHOLD METRICS =====")

    for name, thr in thresholds.items():
        metrics = compute_metrics(gt_array, mse_per_sequence, thr)
        metrics["Threshold_Name"] = name
        all_results.append(metrics)

        print(f"\n➡ Threshold {name} (value={thr:.6f})")
        print(f"AUC={metrics['AUC']:.4f} | Precision={metrics['Precision']:.4f} | Recall={metrics['Recall']:.4f} | "
              f"F1={metrics['F1']:.4f} | FAR={metrics['FAR']:.4f}")

        if best_result is None or metrics["F1"] > best_result["F1"]:
            best_result = metrics

    # ====================== BEST THRESHOLD ======================
    print("\n===== BEST THRESHOLD SELECTED =====")
    print(f"Best Threshold: {best_result['Threshold_Name']} (value={best_result['Threshold_Value']:.6f})")
    print(f"Best F1={best_result['F1']:.4f}, Precision={best_result['Precision']:.4f}, Recall={best_result['Recall']:.4f}")

    # ====================== SAVE SUMMARY ======================
    df_summary = pd.DataFrame(all_results)
    df_summary["Model"] = MODEL_NAME

    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_CALLOUT_MultiThreshold_Summary.csv"
    df_summary.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")
    print("\n✅ LSTM Autoencoder Multi-Threshold Evaluation Completed.")
