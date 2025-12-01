import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from tensorflow.keras import layers, models, regularizers
import tensorflow as tf

tf.keras.utils.set_random_seed(42)

# ======================
# Parameters and Paths
# ======================
MODEL_NAME = "Transformer_Autoencoder"
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
# Sequence Builder
# ======================
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X)


# ======================
# Compute Metrics for Threshold
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
# Transformer Encoder
# ======================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    return x + res


# ======================
# Transformer Autoencoder Model
# ======================
def build_transformer_autoencoder(seq_len, n_features):
    head_size = 16
    num_heads = 2
    ff_dim = 16
    dropout = 0.1

    inputs = layers.Input(shape=(seq_len, n_features))

    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=n_features)(positions)
    x = inputs + pos_emb

    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAvgPool1D(data_format="channels_first")(x)
    encoded = layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l1(1e-4))(x)

    x = layers.Dense(seq_len * n_features)(encoded)
    reconstructed = layers.Reshape((seq_len, n_features))(x)

    model = models.Model(inputs, reconstructed)
    model.compile(optimizer='adam', loss='mse')
    return model


# ======================
# Main Processing
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on ADMISSIONS.csv ===")

    df_raw = pd.read_csv(DATA_PATH)
    X_processed, gt_full = preprocess_admissions(df_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    n_features = X_scaled.shape[1]

    X_seq = create_sequences(X_scaled, SEQUENCE_LENGTH)
    gt_array = gt_full[SEQUENCE_LENGTH - 1:]

    model = build_transformer_autoencoder(SEQUENCE_LENGTH, n_features)
    model.fit(X_seq, X_seq, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    recon = model.predict(X_seq, verbose=0)
    mse = np.mean(np.mean((X_seq - recon) ** 2, axis=2), axis=1)

    mse_reco = mean_squared_error(X_seq.flatten(), recon.flatten())
    r2 = r2_score(X_seq.flatten(), recon.flatten())

    # ======================
    # MULTI-THRESHOLD LOGIC
    # ======================
    thresholds = {
        "90%": np.percentile(mse, 90),
        "95%": np.percentile(mse, 95),
        "98%": np.percentile(mse, 98),
    }

    all_results = []
    best_result = None

    print("\n===== MULTI-THRESHOLD RESULTS =====")
    for name, thr in thresholds.items():
        metrics = compute_metrics(gt_array, mse, thr)
        metrics["Threshold_Name"] = name
        all_results.append(metrics)

        print(f"\n➡ Threshold {name} (Value={thr:.6f})")
        print(f"AUC={metrics['AUC']:.4f} | Precision={metrics['Precision']:.4f} | Recall={metrics['Recall']:.4f} | F1={metrics['F1']:.4f} | FAR={metrics['FAR']:.4f}")

        if best_result is None or metrics["F1"] > best_result["F1"]:
            best_result = metrics

    # ======================
    # PRINT BEST THRESHOLD
    # ======================
    print("\n===== BEST THRESHOLD SELECTED =====")
    print(f"Best = {best_result['Threshold_Name']} (Value={best_result['Threshold_Value']})")
    print(f"Best F1={best_result['F1']:.4f}, Precision={best_result['Precision']:.4f}, Recall={best_result['Recall']:.4f}, FAR={best_result['FAR']:.4f}")


    # ======================
    # SAVE SUMMARY CSV
    # ======================
    df_summary = pd.DataFrame(all_results)
    df_summary["Model"] = MODEL_NAME
    df_summary["Reco_MSE"] = mse_reco
    df_summary["Reco_R2"] = r2

    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_ADMISSIONS_MultiThreshold_Summary.csv"
    df_summary.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")

    # ======================
    # PLOT USING BEST THRESHOLD
    # ======================
    best_thr = best_result["Threshold_Value"]
    y_pred = (mse >= best_thr).astype(int)

    anomalies = X_processed.index[SEQUENCE_LENGTH - 1:][y_pred == 1]

    plt.figure(figsize=(12,6))
    plt.scatter(
        X_processed.index[SEQUENCE_LENGTH - 1:], 
        X_processed['LOS_hours'].iloc[SEQUENCE_LENGTH - 1:],
        c=mse, cmap="viridis", s=20
    )

    plt.scatter(
        anomalies,
        X_processed.loc[anomalies, 'LOS_hours'],
        color="red", edgecolors="red", facecolors="none", s=100,
        label=f"Anomalies ({len(anomalies)})"
    )

    plt.title(f"Transformer AE – Best Threshold ({best_result['Threshold_Name']})")
    plt.xlabel("Record Index")
    plt.ylabel("Length of Stay (Hours)")
    plt.colorbar(label="Reconstruction Error")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ADMISSIONS_best_threshold_plot.png")
    plt.close()

    print("\nTransformer Autoencoder completed successfully.")
