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
SEQUENCE_LENGTH = 10 
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ADMISSIONS.csv")
BASE_OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ADMISSION")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"{MODEL_NAME}-ADMISSIONS-Results"
SUMMARY_DIR = BASE_OUTPUT_DIR / "Summary of Admission"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)


# ======================
# Preprocessing
# ======================
def preprocess_admissions(df):
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df['LOS_hours'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / 3600
    df_filtered = df.dropna(subset=['LOS_hours']).copy()
    features_df = df_filtered[['LOS_hours', 'admission_type', 'discharge_location', 'ethnicity']].copy()
    features_df = pd.get_dummies(features_df, 
                                 columns=['admission_type', 'discharge_location', 'ethnicity'], 
                                 prefix=['adm','disc','eth'], drop_first=True)
    features_df['LOS_hours'].fillna(features_df['LOS_hours'].mean(), inplace=True)
    y_true = df_filtered['hospital_expire_flag'].values
    return features_df, y_true


# ======================
# Sequence Creation
# ======================
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X)


# ======================
# Metrics for Single Threshold
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
# CNN-LSTM Autoencoder Model
# ======================
def build_cnn_lstm_autoencoder(seq_len, n_features):
    model = models.Sequential([
        layers.Input(shape=(seq_len, n_features)),
        layers.Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2, padding='same'),
        layers.LSTM(32, activation='relu', return_sequences=False),
        layers.RepeatVector(seq_len),
        layers.LSTM(32, activation='relu', return_sequences=True),
        layers.TimeDistributed(layers.Dense(n_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ======================
# MAIN
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on ADMISSIONS.csv (Seq Len: {SEQUENCE_LENGTH}) ===")

    df_raw = pd.read_csv(DATA_PATH)
    X_processed, gt_array_full = preprocess_admissions(df_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    n_features = X_scaled.shape[1]

    X_seq = create_sequences(X_scaled, SEQUENCE_LENGTH)
    gt_array = gt_array_full[SEQUENCE_LENGTH - 1:]


    # Train Model
    cnn_lstm_autoencoder = build_cnn_lstm_autoencoder(SEQUENCE_LENGTH, n_features)
    cnn_lstm_autoencoder.fit(X_seq, X_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    # Evaluate
    reconstructions = cnn_lstm_autoencoder.predict(X_seq, verbose=0)
    mse_per_sequence = np.mean(np.mean(np.square(X_seq - reconstructions), axis=2), axis=1)

    mse_reco = mean_squared_error(X_seq.flatten(), reconstructions.flatten())
    r2 = r2_score(X_seq.flatten(), reconstructions.flatten())


    # =====================================================
    # ** MULTI-THRESHOLD EVALUATION **
    # =====================================================
    threshold_values = {
        "90%": np.percentile(mse_per_sequence, 90),
        "95%": np.percentile(mse_per_sequence, 95),
        "98%": np.percentile(mse_per_sequence, 98)
    }

    all_results = []
    best_result = None

    print("\n===== MULTIPLE THRESHOLD RESULTS =====")
    for name, th in threshold_values.items():
        metrics = compute_metrics(gt_array, mse_per_sequence, th)
        metrics["Threshold_Name"] = name
        all_results.append(metrics)

        print(f"\nThreshold {name} (value={th:.6f})")
        print(f"AUC={metrics['AUC']:.4f} | Precision={metrics['Precision']:.4f} | Recall={metrics['Recall']:.4f} | F1={metrics['F1']:.4f} | FAR={metrics['FAR']:.4f}")

        if best_result is None or metrics["F1"] > best_result["F1"]:
            best_result = metrics


    print("\n===== BEST THRESHOLD =====")
    print(f"Best Threshold = {best_result['Threshold_Name']} ({best_result['Threshold_Value']})")
    print(f"F1={best_result['F1']:.4f} | Precision={best_result['Precision']:.4f} | Recall={best_result['Recall']:.4f}\n")


    # =====================================================
    # Save Summary CSV
    # =====================================================
    summary_df = pd.DataFrame(all_results)
    summary_df["Reco_MSE"] = mse_reco
    summary_df["Reco_R2"] = r2
    summary_df["Model"] = MODEL_NAME

    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_ADMISSIONS_MultiThreshold_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"Summary saved → {summary_file}")


    # =====================================================
    # PLOT USING BEST THRESHOLD
    # =====================================================
    best_thr = best_result["Threshold_Value"]
    y_pred_best = (mse_per_sequence >= best_thr).astype(int)
    anomaly_indices = X_processed.index[SEQUENCE_LENGTH - 1:][y_pred_best == 1]

    plt.figure(figsize=(12, 6))
    plt.scatter(X_processed.index[SEQUENCE_LENGTH - 1:], 
                X_processed['LOS_hours'].iloc[SEQUENCE_LENGTH - 1:], 
                c=mse_per_sequence, cmap='viridis', s=20)

    plt.scatter(anomaly_indices,
                X_processed.loc[anomaly_indices, 'LOS_hours'],
                color='red', edgecolors='red', facecolors='none', s=100,
                label=f"Detected Anomalies ({len(anomaly_indices)})")

    plt.title(f"{MODEL_NAME} - Anomalies using BEST Threshold ({best_result['Threshold_Name']})")
    plt.xlabel("Record Index")
    plt.ylabel("Length of Stay (Hours)")
    plt.colorbar(label="Reconstruction Error")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ADMISSIONS_best_threshold_plot.png")
    plt.close()

    print("\nCNN-LSTM Autoencoder Multi-Threshold Evaluation Completed Successfully.")
