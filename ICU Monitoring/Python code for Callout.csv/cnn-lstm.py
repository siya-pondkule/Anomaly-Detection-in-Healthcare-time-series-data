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
MODEL_NAME = "CNN_LSTM_Autoencoder"
SEQUENCE_LENGTH = 5 
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\CALLOUT.csv")
BASE_OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for CALLOUT")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"{MODEL_NAME}-CALLOUT-Results"
SUMMARY_DIR = BASE_OUTPUT_DIR / "Summary of Callout"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)


# ======================
# Preprocessing (unchanged)
# ======================
def preprocess_callouts(df):
    df['createtime'] = pd.to_datetime(df['createtime'])
    df['outcometime'] = pd.to_datetime(df['outcometime'])
    df['acknowledgetime'] = pd.to_datetime(df['acknowledgetime'])
    
    df['duration_hours'] = (df['outcometime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'] = (df['acknowledgetime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'].fillna(df['ack_lag_hours'].mean(), inplace=True)
    
    features_df = df[['duration_hours','ack_lag_hours','request_tele','request_resp','request_cdiff',
                      'request_mrsa','request_vre','curr_careunit','callout_service','acknowledge_status']].copy()

    features_df = pd.get_dummies(features_df, 
                                 columns=['curr_careunit','callout_service','acknowledge_status'], 
                                 drop_first=True)

    y_true = (df['callout_outcome'] == 'Cancelled').astype(int).values
    return features_df, y_true


# ======================
# Sequence Creator (unchanged)
# ======================
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X)


# ======================
# Metrics for one threshold
# ======================
def compute_metrics(y_true, scores, threshold):
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
# CNN-LSTM Autoencoder (unchanged)
# ======================
def build_cnn_lstm_autoencoder(seq_len, n_features):
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2, padding='same'),
        layers.LSTM(8, activation='relu', return_sequences=False),
        layers.RepeatVector(seq_len),
        layers.LSTM(8, activation='relu', return_sequences=True),
        layers.TimeDistributed(layers.Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ======================
# Main Processing
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on CALLOUT.csv (Seq={SEQUENCE_LENGTH}) ===")
    
    df_raw = pd.read_csv(DATA_PATH)
    X_processed, gt_array_full = preprocess_callouts(df_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    n_features = X_scaled.shape[1]

    X_seq = create_sequences(X_scaled, SEQUENCE_LENGTH)
    gt_array = gt_array_full[SEQUENCE_LENGTH - 1:]

    # Train model
    model = build_cnn_lstm_autoencoder(SEQUENCE_LENGTH, n_features)
    model.fit(X_seq, X_seq, epochs=50, batch_size=4, validation_split=0.1, verbose=0)

    # Predictions
    recon = model.predict(X_seq, verbose=0)
    mse_scores = np.mean(np.mean(np.square(X_seq - recon), axis=2), axis=1)

    mse_reco = mean_squared_error(X_seq.flatten(), recon.flatten())
    r2 = r2_score(X_seq.flatten(), recon.flatten())


    # ======================
    # MULTI-THRESHOLD PROCESSING
    # ======================
    thresholds = {
        "90%": np.percentile(mse_scores, 90),
        "95%": np.percentile(mse_scores, 95),
        "98%": np.percentile(mse_scores, 98)
    }

    all_results = []
    best_threshold_result = None

    print("\n===== MULTI-THRESHOLD RESULTS =====")

    for name, thr in thresholds.items():
        result = compute_metrics(gt_array, mse_scores, thr)
        result["Threshold_Name"] = name
        all_results.append(result)

        print(f"\n➡ Threshold: {name} (value={thr:.6f})")
        print(f"AUC={result['AUC']:.4f} | Precision={result['Precision']:.4f} | Recall={result['Recall']:.4f} | F1={result['F1']:.4f} | FAR={result['FAR']:.4f}")

        # Best = highest F1
        if best_threshold_result is None or result["F1"] > best_threshold_result["F1"]:
            best_threshold_result = result


    # ======================
    # BEST THRESHOLD
    # ======================
    print("\n===== BEST THRESHOLD SELECTED =====")
    print(f"Best = {best_threshold_result['Threshold_Name']} (Value={best_threshold_result['Threshold_Value']:.6f})")
    print(f"Best F1={best_threshold_result['F1']:.4f}, Precision={best_threshold_result['Precision']:.4f}, Recall={best_threshold_result['Recall']:.4f}")


    # ======================
    # SAVE MULTI-THRESHOLD SUMMARY CSV
    # ======================
    df_summary = pd.DataFrame(all_results)
    df_summary["Model"] = MODEL_NAME
    df_summary["Reco_MSE"] = mse_reco
    df_summary["Reco_R2"] = r2

    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_CALLOUT_MultiThreshold_Summary.csv"
    df_summary.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")


    # ======================
    # PLOT using BEST THRESHOLD
    # ======================
    best_thr = best_threshold_result["Threshold_Value"]
    y_pred_best = (mse_scores >= best_thr).astype(int)

    plot_indices = X_processed.index[SEQUENCE_LENGTH - 1:]
    anomalies = plot_indices[y_pred_best == 1]

    plt.figure(figsize=(12,6))
    plt.scatter(plot_indices,
                X_processed['duration_hours'].iloc[SEQUENCE_LENGTH - 1:],
                c=mse_scores, cmap='viridis', s=20)

    plt.scatter(anomalies,
                X_processed.loc[anomalies, 'duration_hours'],
                color="red", edgecolors="red", facecolors="none", s=100,
                label=f"Predicted Anomalies ({len(anomalies)})")

    plt.title(f"CALLOUT CNN-LSTM AE (Best Threshold: {best_threshold_result['Threshold_Name']})")
    plt.xlabel("Record Index")
    plt.ylabel("Callout Duration (Hours)")
    plt.colorbar(label='Reconstruction Error')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "CALLOUT_best_threshold_plot.png")
    plt.close()

    print("\n✅ CNN-LSTM Autoencoder completed successfully.")
