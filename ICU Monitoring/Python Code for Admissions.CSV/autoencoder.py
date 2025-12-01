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

# ======================
# Paths and Parameters
# ======================
MODEL_NAME = "Autoencoder"
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ADMISSIONS.csv")
BASE_OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ADMISSION")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"{MODEL_NAME}-ADMISSIONS-Results"
SUMMARY_DIR = BASE_OUTPUT_DIR / "Summary of Admission"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)


# ======================
# Preprocessing Function
# ======================
def preprocess_admissions(df):
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df['LOS_hours'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / 3600

    features_df = df[['LOS_hours', 'admission_type', 'discharge_location', 'ethnicity']].copy()

    features_df = pd.get_dummies(
        features_df,
        columns=['admission_type', 'discharge_location', 'ethnicity'],
        drop_first=True
    )

    features_df['LOS_hours'].fillna(features_df['LOS_hours'].mean(), inplace=True)

    y_true = df['hospital_expire_flag'].values

    return features_df, y_true


# ======================
# Autoencoder Model
# ======================
def build_autoencoder(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu', name='encoded'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# ======================
# Metric Computation for a given threshold
# ======================
def compute_metrics_for_threshold(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_true, scores)
    except:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "Threshold": threshold,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FAR": far
    }


# ======================
# Main Processing
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on ADMISSIONS.csv ===")

    df_raw = pd.read_csv(DATA_PATH)
    X_processed, gt_array = preprocess_admissions(df_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    input_dim = X_scaled.shape[1]

    autoencoder = build_autoencoder(input_dim)
    history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    print(f"Training complete | Final Loss: {history.history['loss'][-1]:.4f}")

    reconstructions = autoencoder.predict(X_scaled, verbose=0)
    mse_per_row = np.mean(np.square(X_scaled - reconstructions), axis=1)

    # =====================================================
    # MULTI-THRESHOLD EVALUATION (90th, 95th, 98th percentile)
    # =====================================================
    thresholds = {
        "90%": np.percentile(mse_per_row, 90),
        "95%": np.percentile(mse_per_row, 95),
        "98%": np.percentile(mse_per_row, 98)
    }

    results_list = []
    best_result = None

    for name, th in thresholds.items():
        metrics = compute_metrics_for_threshold(gt_array, mse_per_row, th)
        metrics["Threshold_Name"] = name
        results_list.append(metrics)

        if best_result is None or metrics["F1"] > best_result["F1"]:
            best_result = metrics

    print("\n===== MULTI-THRESHOLD RESULTS =====")
    for res in results_list:
        print(f"\nThreshold {res['Threshold_Name']} (value={res['Threshold']:.6f}):")
        print(f"AUC={res['AUC']:.4f} | Precision={res['Precision']:.4f} | Recall={res['Recall']:.4f} | "
              f"F1={res['F1']:.4f} | FAR={res['FAR']:.4f}")

    # Reconstruction quality
    mse_reco = mean_squared_error(X_scaled, reconstructions)
    r2 = r2_score(X_scaled, reconstructions)

    # ======================
    # Save Summary
    # ======================
    summary_df = pd.DataFrame(results_list)
    summary_df["Reco_MSE"] = mse_reco
    summary_df["Reco_R2"] = r2
    summary_df["Model"] = MODEL_NAME

    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_ADMISSIONS_MultiThreshold_Summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\nSummary saved → {summary_file}")

    print("\n===== BEST THRESHOLD =====")
    print(f"Best Threshold: {best_result['Threshold_Name']} ({best_result['Threshold']})")
    print(f"Best F1: {best_result['F1']:.4f}")
    print(f"Precision={best_result['Precision']:.4f} | Recall={best_result['Recall']:.4f} | FAR={best_result['FAR']:.4f}")

    # ======================
    # Plot (using the best threshold)
    # ======================
    best_thr = best_result["Threshold"]
    y_pred_model = (mse_per_row >= best_thr).astype(int)
    model_anomalies_indices = X_processed.index[y_pred_model == 1]

    plt.figure(figsize=(12, 6))
    plt.scatter(X_processed.index, X_processed['LOS_hours'],
                c=mse_per_row, cmap='viridis', s=20, label='LOS (Error Colored)')

    plt.scatter(model_anomalies_indices,
                X_processed.loc[model_anomalies_indices, 'LOS_hours'],
                color="red", edgecolors='red', facecolors='none', s=100,
                label=f'Anomalies by Best Threshold ({len(model_anomalies_indices)})')

    plt.title(f"ADMISSIONS Anomaly Detection ({MODEL_NAME}) - Using Best Threshold")
    plt.xlabel("Record Index")
    plt.ylabel("Length of Stay (Hours)")
    plt.colorbar(label='Reconstruction Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"ADMISSIONS_anomaly_plot_best_threshold.png")
    plt.close()

    print("\nAutoencoder Multi-threshold Evaluation Completed Successfully.")
