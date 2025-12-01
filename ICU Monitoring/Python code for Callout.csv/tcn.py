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

tf.keras.utils.set_random_seed(42)

# ======================
# Parameters and Paths
# ======================
MODEL_NAME = "TCN_Autoencoder"
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

    features_df = df[['duration_hours', 'ack_lag_hours',
                      'request_tele', 'request_resp', 'request_cdiff',
                      'request_mrsa', 'request_vre',
                      'curr_careunit', 'callout_service', 'acknowledge_status']].copy()

    features_df = pd.get_dummies(features_df, 
                                 columns=['curr_careunit', 'callout_service', 'acknowledge_status'], 
                                 drop_first=True)

    y_true = (df['callout_outcome'] == 'Cancelled').astype(int).values
    return features_df, y_true

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X)


# ======================
# Multi-threshold evaluation
# ======================
def evaluate_thresholds(y_true, mse_scores):
    thresholds = {
        "90%": np.percentile(mse_scores, 90),
        "95%": np.percentile(mse_scores, 95),
        "98%": np.percentile(mse_scores, 98)
    }

    results = []
    best_result = None

    for name, thr in thresholds.items():
        y_pred = (mse_scores >= thr).astype(int)

        try:
            auc = roc_auc_score(y_true, mse_scores)
        except:
            auc = np.nan

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0

        res = {
            "Threshold": name,
            "Threshold_Value": thr,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "FAR": far
        }
        results.append(res)

        if best_result is None or f1 > best_result["F1"]:
            best_result = res

    return results, best_result


# ======================
# TCN Model Definition
# ======================
def build_tcn_autoencoder(seq_len, n_features):
    input_seq = layers.Input(shape=(seq_len, n_features))

    h = layers.Conv1D(filters=16, kernel_size=2, activation='relu', padding='causal')(input_seq)
    h = layers.Conv1D(filters=8, kernel_size=2, activation='relu', padding='causal')(h)

    compressed = layers.Flatten()(h)
    compressed = layers.Dense(4, activation='relu')(compressed)

    h = layers.RepeatVector(seq_len)(compressed)
    h = layers.Conv1D(filters=8, kernel_size=2, activation='relu', padding='causal')(h)
    h = layers.Conv1D(filters=16, kernel_size=2, activation='relu', padding='causal')(h)

    output_seq = layers.Conv1D(filters=n_features, kernel_size=1, activation='linear')(h)

    model = models.Model(inputs=input_seq, outputs=output_seq)
    model.compile(optimizer='adam', loss='mse')
    return model


# ======================
# Main
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

    tcn_autoencoder = build_tcn_autoencoder(SEQUENCE_LENGTH, n_features)
    tcn_autoencoder.fit(X_seq, X_seq, epochs=50, batch_size=4, validation_split=0.1, verbose=0)

    reconstructions = tcn_autoencoder.predict(X_seq, verbose=0)
    mse_per_sequence = np.mean(np.mean(np.square(X_seq - reconstructions), axis=2), axis=1)

    # ---- MULTI-THRESHOLD EVALUATION ----
    all_results, best_result = evaluate_thresholds(gt_array, mse_per_sequence)

    print("\n===== MULTI-THRESHOLD RESULTS =====")
    for r in all_results:
        print(f"\n{r}")

    print("\n===== BEST THRESHOLD =====")
    print(best_result)

    # ---- SAVE SUMMARY ----
    df_summary = pd.DataFrame(all_results)
    df_summary["Model"] = MODEL_NAME

    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_CALLOUT_MultiThreshold_Summary.csv"
    df_summary.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")
    print("✅ Completed successfully")
