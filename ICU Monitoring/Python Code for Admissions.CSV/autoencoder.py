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
# Data Preprocessing and Feature Engineering
# ======================
def preprocess_admissions(df):
    # Convert timestamps to datetime
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    
    # Calculate Length of Stay (LOS) in hours
    df['LOS_hours'] = (df['dischtime'] - df['admittime']).dt.total_seconds() / 3600
    
    # Features for Anomaly Detection
    features_df = df[['LOS_hours', 'admission_type', 'discharge_location', 'ethnicity']].copy()
    
    # One-Hot Encode categorical features
    features_df = pd.get_dummies(
        features_df, 
        columns=['admission_type', 'discharge_location', 'ethnicity'], 
        drop_first=True
    )
    
    # Handle NaN values in LOS_hours (e.g., replace with mean or 0)
    features_df['LOS_hours'].fillna(features_df['LOS_hours'].mean(), inplace=True)
    
    # Ground Truth: Hospital Expire Flag (1 = anomaly/adverse outcome)
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
# Evaluation Function (Finds best threshold for F1-score)
# ======================
def evaluate_anomaly_detection(y_true, mse_scores):
    try:
        auc = roc_auc_score(y_true, mse_scores)
    except ValueError:
        auc = np.nan

    best_f1, best_threshold = 0, np.percentile(mse_scores, 90)
    
    if np.sum(y_true) == 0:
        return {
            "AUC": auc, "F1": np.nan, "Precision": np.nan, 
            "Recall": np.nan, "FAR": np.nan, "Threshold": best_threshold
        }

    # Iterate over high percentiles to find the optimal threshold
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

    return {
        "AUC": auc,
        "F1": best_f1,
        "Precision": precision_score(y_true, y_pred_best, zero_division=0),
        "Recall": recall_score(y_true, y_pred_best, zero_division=0),
        "FAR": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "Threshold": best_threshold
    }


# ======================
# Main Processing
# ======================
if __name__ == "__main__":
    print(f"\n=== Running {MODEL_NAME} on ADMISSIONS.csv ===")
    
    # Load and Preprocess Data
    df_raw = pd.read_csv(DATA_PATH)
    X_processed, gt_array = preprocess_admissions(df_raw)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)
    input_dim = X_scaled.shape[1]

    # Train Autoencoder
    autoencoder = build_autoencoder(input_dim)
    history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
    print(f"Training complete | Final Loss: {history.history['loss'][-1]:.4f}")

    # Reconstruction and evaluation
    reconstructions = autoencoder.predict(X_scaled, verbose=0)
    # MSE is the anomaly score
    mse_per_row = np.mean(np.square(X_scaled - reconstructions), axis=1)
    
    # Evaluate using the 'hospital_expire_flag' as ground truth
    eval_results = evaluate_anomaly_detection(gt_array, mse_per_row)

    print(f"\nResults for ADMISSIONS.csv ({MODEL_NAME}):")
    print(f"Ground Truth (Adverse Outcome): {np.sum(gt_array)} records out of {len(gt_array)}")
    print(f"AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | "
          f"Precision={eval_results['Precision']:.4f} | Recall={eval_results['Recall']:.4f} | "
          f"FAR={eval_results['FAR']:.4f}")

    # Reconstruction performance
    mse_reco = mean_squared_error(X_scaled, reconstructions)
    r2 = r2_score(X_scaled, reconstructions)
    print(f"Reconstruction: MSE={mse_reco:.4f} | R²={r2:.4f}")

    # ======================
    # Summary: Save Evaluation
    # ======================
    summary_data = {
        "Model": [MODEL_NAME],
        "AUC": [eval_results['AUC']],
        "F1": [eval_results['F1']],
        "Precision": [eval_results['Precision']],
        "Recall": [eval_results['Recall']],
        "FAR": [eval_results['FAR']],
        "Reco_MSE": [mse_reco],
        "Reco_R2": [r2]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = SUMMARY_DIR / f"{MODEL_NAME}_ADMISSIONS_Evaluation_Summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nEvaluation summary saved → {summary_file}")

    # ======================
    # Plot anomalies (Focus on LOS)
    # ======================
    plt.figure(figsize=(12,6))
    plt.scatter(X_processed.index, X_processed['LOS_hours'], 
                c=mse_per_row, cmap='viridis', s=20, label='LOS (Color by Error)')
    
    # Identify model-detected anomalies based on best threshold
    y_pred_model = (mse_per_row >= eval_results['Threshold']).astype(int)
    model_anomalies_indices = X_processed.index[y_pred_model == 1]
    
    # Mark the predicted anomalies
    plt.scatter(model_anomalies_indices, X_processed.loc[model_anomalies_indices, 'LOS_hours'], 
                color="red", marker="o", edgecolors='red', facecolors='none', s=100, 
                label=f'Predicted Anomalies ({len(model_anomalies_indices)})')

    plt.title(f"ADMISSIONS Anomaly Detection ({MODEL_NAME}): LOS colored by Reconstruction Error")
    plt.xlabel("Record Index")
    plt.ylabel("Length of Stay (Hours)")
    plt.colorbar(label='Reconstruction Error (Anomaly Score)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ADMISSIONS_anomaly_plot_LOS.png")
    plt.close()

    print(f"✅ {MODEL_NAME} Model completed successfully.")