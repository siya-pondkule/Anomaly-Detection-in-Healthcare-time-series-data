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

# (Reused Preprocessing and Evaluation functions)
def preprocess_callouts(df):
    # FIX: Changed pd.to_to_datetime to pd.to_datetime
    df['createtime'] = pd.to_datetime(df['createtime'])
    df['outcometime'] = pd.to_datetime(df['outcometime'])
    df['acknowledgetime'] = pd.to_datetime(df['acknowledgetime'])
    df['duration_hours'] = (df['outcometime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'] = (df['acknowledgetime'] - df['createtime']).dt.total_seconds() / 3600
    df['ack_lag_hours'].fillna(df['ack_lag_hours'].mean(), inplace=True)
    features_df = df[['duration_hours', 'ack_lag_hours', 'request_tele', 'request_resp', 'request_cdiff', 'request_mrsa', 'request_vre', 'curr_careunit', 'callout_service', 'acknowledge_status']].copy()
    features_df = pd.get_dummies(features_df, columns=['curr_careunit', 'callout_service', 'acknowledge_status'], drop_first=True)
    y_true = (df['callout_outcome'] == 'Cancelled').astype(int).values
    return features_df, y_true

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X)

def evaluate_anomaly_detection(y_true, mse_scores):
    try:
        auc = roc_auc_score(y_true, mse_scores)
    except ValueError:
        auc = np.nan
    best_f1, best_threshold = 0, np.percentile(mse_scores, 95) 
    if np.sum(y_true) == 0:
        return {"AUC": auc, "F1": np.nan, "Precision": np.nan, "Recall": np.nan, "FAR": np.nan, "Threshold": best_threshold}
    for q in np.linspace(90, 99.9, 100):
        threshold = np.percentile(mse_scores, q)
        y_pred = (mse_scores >= threshold).astype(int)
        if np.sum(y_pred) > 0:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
    y_pred_best = (mse_scores >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {"AUC": auc, "F1": best_f1, "Precision": precision_score(y_true, y_pred_best, zero_division=0), "Recall": recall_score(y_true, y_pred_best, zero_division=0), "FAR": fp / (fp + tn) if (fp + tn) > 0 else 0, "Threshold": best_threshold}


# ======================
# Model Definition (CNN-LSTM Autoencoder)
# ======================
def build_cnn_lstm_autoencoder(seq_len, n_features):
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        # Encoder: CNN for feature extraction
        layers.Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2, padding='same'),
        # Encoder: LSTM for sequence learning
        layers.LSTM(8, activation='relu', return_sequences=False),
        
        layers.RepeatVector(seq_len), # Bottleneck
        
        # Decoder: LSTM
        layers.LSTM(8, activation='relu', return_sequences=True),
        # Decoder: TimeDistributed Dense
        layers.TimeDistributed(layers.Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ======================
# Main Processing
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on CALLOUT.csv (Seq Len: {SEQUENCE_LENGTH}) ===")
    
    df_raw = pd.read_csv(DATA_PATH)
    X_processed, gt_array_full = preprocess_callouts(df_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    n_features = X_scaled.shape[1]

    X_seq = create_sequences(X_scaled, SEQUENCE_LENGTH)
    gt_array = gt_array_full[SEQUENCE_LENGTH - 1:]

    # Train Model
    cnn_lstm_autoencoder = build_cnn_lstm_autoencoder(SEQUENCE_LENGTH, n_features)
    history = cnn_lstm_autoencoder.fit(X_seq, X_seq, epochs=50, batch_size=4, validation_split=0.1, verbose=0)
    
    # Evaluate
    reconstructions = cnn_lstm_autoencoder.predict(X_seq, verbose=0)
    mse_per_sequence = np.mean(np.mean(np.square(X_seq - reconstructions), axis=2), axis=1)
    eval_results = evaluate_anomaly_detection(gt_array, mse_per_sequence)

    mse_reco = mean_squared_error(X_seq.flatten(), reconstructions.flatten())
    r2 = r2_score(X_seq.flatten(), reconstructions.flatten())

    print(f"\nResults for CALLOUT.csv ({MODEL_NAME}):")
    print(f"Ground Truth ('Cancelled' outcome): {np.sum(gt_array)} records out of {len(gt_array)}")
    print(f"AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | Reco_MSE={mse_reco:.4f}")

    # Save Summary
    summary_data = {"Model": [MODEL_NAME], "AUC": [eval_results['AUC']], "F1": [eval_results['F1']], "Precision": [eval_results['Precision']], "Recall": [eval_results['Recall']], "FAR": [eval_results['FAR']], "Reco_MSE": [mse_reco], "Reco_R2": [r2]}
    summary_df = pd.DataFrame(summary_data)
    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_CALLOUT_Evaluation_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    # Plot
    plt.figure(figsize=(12,6))
    duration_data = X_processed['duration_hours'].iloc[SEQUENCE_LENGTH-1:]
    plot_indices = X_processed.index[SEQUENCE_LENGTH-1:]
    plt.scatter(plot_indices, duration_data, c=mse_per_sequence, cmap='viridis', s=20, label='Duration (Color by Error)')
    y_pred_model = (mse_per_sequence >= eval_results['Threshold']).astype(int)
    model_anomalies_indices = plot_indices[y_pred_model == 1]
    plt.scatter(model_anomalies_indices, X_processed.loc[model_anomalies_indices, 'duration_hours'], color="red", marker="o", edgecolors='red', facecolors='none', s=100, label=f'Predicted Anomalies ({len(model_anomalies_indices)})')
    plt.title(f"CALLOUT Anomaly Detection ({MODEL_NAME}): Callout Duration colored by Reconstruction Error")
    plt.xlabel("Record Index")
    plt.ylabel("Callout Duration (Hours)")
    plt.colorbar(label='Reconstruction Error (Anomaly Score)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "CALLOUT_anomaly_plot_duration.png")
    plt.close()

    print(f"✅ {MODEL_NAME} Model completed successfully. Results saved to → {OUTPUT_DIR}")