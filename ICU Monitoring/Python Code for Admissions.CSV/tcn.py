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
SEQUENCE_LENGTH = 10 
DATA_PATH = Path(r"D:\\Final Year\\Project\\Anomaly Detection\\Anomaly Detection\\ICU Monitoring\\Datasets\\ADMISSIONS.csv")
BASE_OUTPUT_DIR = Path(r"D:\\Final Year\\Project\\Anomaly Detection\\Anomaly Detection\\ICU Monitoring\\Results of all Models for ADMISSION")
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
    features_df = pd.get_dummies(
        features_df,
        columns=['admission_type','discharge_location','ethnicity'],
        prefix=['adm','disc','eth'],
        drop_first=True
    )
    features_df['LOS_hours'].fillna(features_df['LOS_hours'].mean(), inplace=True)

    y_true = df_filtered['hospital_expire_flag'].values
    return features_df, y_true


# ======================
# Create Sequences
# ======================
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X)


# ======================
# Metrics for Threshold
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
# TCN Autoencoder
# ======================
def build_tcn_autoencoder(seq_len, n_features):
    inp = layers.Input(shape=(seq_len, n_features))

    # Encoder
    h = layers.Conv1D(32, 2, activation='relu', padding='causal')(inp)
    h = layers.Conv1D(16, 2, activation='relu', padding='causal')(h)

    compressed = layers.Flatten()(h)
    compressed = layers.Dense(8, activation='relu')(compressed)

    # Decoder
    h = layers.RepeatVector(seq_len)(compressed)
    h = layers.Conv1D(16, 2, activation='relu', padding='causal')(h)
    h = layers.Conv1D(32, 2, activation='relu', padding='causal')(h)

    out = layers.Conv1D(n_features, 1, activation='linear')(h)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model


# ======================
# Main Processing
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on ADMISSIONS.csv ===")

    df_raw = pd.read_csv(DATA_PATH)
    X_processed, gt_array_full = preprocess_admissions(df_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    n_features = X_scaled.shape[1]
    X_seq = create_sequences(X_scaled, SEQUENCE_LENGTH)
    gt_array = gt_array_full[SEQUENCE_LENGTH - 1:]

    tcn_autoencoder = build_tcn_autoencoder(SEQUENCE_LENGTH, n_features)
    tcn_autoencoder.fit(X_seq, X_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    recon = tcn_autoencoder.predict(X_seq, verbose=0)
    mse_scores = np.mean(np.mean(np.square(X_seq - recon), axis=2), axis=1)

    mse_reco = mean_squared_error(X_seq.flatten(), recon.flatten())
    r2 = r2_score(X_seq.flatten(), recon.flatten())

    # =====================================================
    # MULTI-THRESHOLD EVALUATION (90, 95, 98)
    # =====================================================
    threshold_values = {
        "90%": np.percentile(mse_scores, 90),
        "95%": np.percentile(mse_scores, 95),
        "98%": np.percentile(mse_scores, 98)
    }

    all_results = []
    best_result = None

    print("\n===== MULTI-THRESHOLD RESULTS =====")
    for name, th in threshold_values.items():
        metrics = compute_metrics(gt_array, mse_scores, th)
        metrics["Threshold_Name"] = name
        all_results.append(metrics)

        print(f"\nThreshold {name}  (Value={th:.6f})")
        print(f"AUC={metrics['AUC']:.4f} | Precision={metrics['Precision']:.4f} | "
              f"Recall={metrics['Recall']:.4f} | F1={metrics['F1']:.4f} | FAR={metrics['FAR']:.4f}")

        if best_result is None or metrics["F1"] > best_result["F1"]:
            best_result = metrics


    # ========================
    # BEST THRESHOLD SUMMARY
    # ========================
    print("\n===== BEST THRESHOLD SELECTED =====")
    print(f"Best Threshold = {best_result['Threshold_Name']} (Value={best_result['Threshold_Value']})")
    print(f"Best F1 = {best_result['F1']:.4f}")
    print(f"Precision={best_result['Precision']:.4f} | Recall={best_result['Recall']:.4f} | FAR={best_result['FAR']:.4f}")


    # ======================
    # Save Summary CSV
    # ======================
    summary_df = pd.DataFrame(all_results)
    summary_df["Model"] = MODEL_NAME
    summary_df["Reco_MSE"] = mse_reco
    summary_df["Reco_R2"] = r2

    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_ADMISSIONS_MultiThreshold_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")


    # ======================
    # Plot Using Best Threshold
    # ======================
    best_thr = best_result["Threshold_Value"]
    y_pred_best = (mse_scores >= best_thr).astype(int)

    anomaly_indices = X_processed.index[SEQUENCE_LENGTH - 1:][y_pred_best == 1]

    plt.figure(figsize=(12,6))
    plt.scatter(X_processed.index[SEQUENCE_LENGTH-1:], 
                X_processed['LOS_hours'].iloc[SEQUENCE_LENGTH-1:], 
                c=mse_scores, cmap='viridis', s=20)

    plt.scatter(
        anomaly_indices,
        X_processed.loc[anomaly_indices, 'LOS_hours'],
        color="red", edgecolors="red", facecolors="none", s=100,
        label=f"Detected Anomalies ({len(anomaly_indices)})"
    )

    plt.title(f"{MODEL_NAME} - LOS Anomalies using Best Threshold ({best_result['Threshold_Name']})")
    plt.xlabel("Record Index")
    plt.ylabel("Length of Stay (Hours)")
    plt.colorbar(label="Reconstruction Error")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ADMISSIONS_best_threshold_plot.png")
    plt.close()

    print(f"\n{MODEL_NAME} Model completed successfully.")
