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
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ADMISSIONS.csv")
BASE_OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ADMISSION")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"{MODEL_NAME}-ADMISSIONS-Results"
SUMMARY_DIR = BASE_OUTPUT_DIR / "Summary of Admission"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# (Preprocessing and Evaluation functions are identical to LSTM-AE script)

def preprocess_admissions(df):
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df['LOS_hours'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / 3600
    df_filtered = df.dropna(subset=['LOS_hours']).copy()
    features_df = df_filtered[['LOS_hours', 'admission_type', 'discharge_location', 'ethnicity']].copy()
    features_df = pd.get_dummies(features_df, columns=['admission_type', 'discharge_location', 'ethnicity'], prefix=['adm', 'disc', 'eth'], drop_first=True)
    features_df['LOS_hours'].fillna(features_df['LOS_hours'].mean(), inplace=True)
    y_true = df_filtered['hospital_expire_flag'].values
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
    best_f1, best_threshold = 0, np.percentile(mse_scores, 90)
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
# Model Definition (TCN-like Autoencoder)
# ======================
def build_tcn_autoencoder(seq_len, n_features):
    input_seq = layers.Input(shape=(seq_len, n_features))
    
    # Encoder (Causal padding for TCN style)
    h = layers.Conv1D(filters=32, kernel_size=2, activation='relu', padding='causal')(input_seq)
    h = layers.Conv1D(filters=16, kernel_size=2, activation='relu', padding='causal')(h)
    
    # Bottleneck
    compressed = layers.Flatten()(h)
    compressed = layers.Dense(8, activation='relu')(compressed)
    
    # Decoder 
    h = layers.RepeatVector(seq_len)(compressed)
    
    h = layers.Conv1D(filters=16, kernel_size=2, activation='relu', padding='causal')(h)
    h = layers.Conv1D(filters=32, kernel_size=2, activation='relu', padding='causal')(h)
    
    # Output layer
    output_seq = layers.Conv1D(filters=n_features, kernel_size=1, activation='linear')(h)

    model = models.Model(inputs=input_seq, outputs=output_seq)
    model.compile(optimizer='adam', loss='mse')
    return model


# ======================
# Main Processing
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
    tcn_autoencoder = build_tcn_autoencoder(SEQUENCE_LENGTH, n_features)
    history = tcn_autoencoder.fit(X_seq, X_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    
    # Evaluate
    reconstructions = tcn_autoencoder.predict(X_seq, verbose=0)
    mse_per_sequence = np.mean(np.mean(np.square(X_seq - reconstructions), axis=2), axis=1)
    eval_results = evaluate_anomaly_detection(gt_array, mse_per_sequence)

    mse_reco = mean_squared_error(X_seq.flatten(), reconstructions.flatten())
    r2 = r2_score(X_seq.flatten(), reconstructions.flatten())

    print(f"\nResults for ADMISSIONS.csv ({MODEL_NAME}):")
    print(f"AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | Reco_MSE={mse_reco:.4f}")

    # Save Summary
    summary_data = {"Model": [MODEL_NAME], "AUC": [eval_results['AUC']], "F1": [eval_results['F1']], "Precision": [eval_results['Precision']], "Recall": [eval_results['Recall']], "FAR": [eval_results['FAR']], "Reco_MSE": [mse_reco], "Reco_R2": [r2]}
    summary_df = pd.DataFrame(summary_data)
    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_ADMISSIONS_Evaluation_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    # Plot
    plt.figure(figsize=(12,6))
    plt.scatter(X_processed.index[SEQUENCE_LENGTH-1:], X_processed['LOS_hours'].iloc[SEQUENCE_LENGTH-1:], c=mse_per_sequence, cmap='viridis', s=20, label='LOS (Color by Error)')
    y_pred_model = (mse_per_sequence >= eval_results['Threshold']).astype(int)
    model_anomalies_indices = X_processed.index[SEQUENCE_LENGTH-1:][y_pred_model == 1]
    plt.scatter(model_anomalies_indices, X_processed.loc[model_anomalies_indices, 'LOS_hours'], color="red", marker="o", edgecolors='red', facecolors='none', s=100, label=f'Predicted Anomalies ({len(model_anomalies_indices)})')
    plt.title(f"ADMISSIONS Anomaly Detection ({MODEL_NAME}): LOS colored by Reconstruction Error")
    plt.xlabel("Record Index")
    plt.ylabel("Length of Stay (Hours)")
    plt.colorbar(label='Reconstruction Error (Anomaly Score)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ADMISSIONS_anomaly_plot_LOS.png")
    plt.close()

    print(f"✅ {MODEL_NAME} Model completed successfully. Results saved to → {OUTPUT_DIR}")