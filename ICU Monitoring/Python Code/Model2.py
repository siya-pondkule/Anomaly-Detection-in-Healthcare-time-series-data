# ICU anomaly detection + correction +  mortality prediction

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    brier_score_loss, precision_recall_curve, auc, classification_report
)
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------
# TensorFlow optional
# ----------------------
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    TF_AVAILABLE = False
    tf = None
    layers = None
    models = None

# ----------------------
# Config / Paths
# ----------------------
DATASETS_DIR = Path(r"d:\Anomaly Detection\ICU Monitoring\Datasets")
OUTPUT_DIR = Path(DATASETS_DIR) / "../Model2Results"
OUTPUT_DIR = OUTPUT_DIR.resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TH = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90,
    "DBP_high": 90, "DBP_low": 60,
    "MAP_high": 110, "MAP_low": 70,
    "RR_tachypnea": 24, "RR_apnea": 8,
    "SpO2_low": 90,
    "Temp_high": 38.5, "Temp_low": 35.0
}

VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR", "PR"],
    "SBP": ["SysBP", "SBP", "SystolicBP", "SYS"],
    "DBP": ["DiaBP", "DBP", "DiastolicBP", "DIA"],
    "MAP": ["MAP", "MeanBP", "MeanArterialPressure"],
    "RR": ["RespRate", "RR", "Resp", "RespiratoryRate"],
    "SpO2": ["SpO2", "OxygenSaturation", "O2Sat", "SaO2"],
    "Temp": ["Temp", "Temperature", "BodyTemp", "T"]
}

# ----------------------
# Helper Functions
# ----------------------
def canonical_vital(col_name):
    for v, cols in VITAL_MAPPING.items():
        if col_name in cols:
            return v
    return None

def detect_anomalies(series, col_name):
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    idxs = []
    vital = canonical_vital(col_name)
    if vital:
        if vital == "HR":
            idxs += list(series[series > TH["HR_tachy"]].index)
            idxs += list(series[series < TH["HR_brady"]].index)
        elif vital == "SBP":
            idxs += list(series[series > TH["SBP_high"]].index)
            idxs += list(series[series < TH["SBP_low"]].index)
        elif vital == "DBP":
            idxs += list(series[series > TH["DBP_high"]].index)
            idxs += list(series[series < TH["DBP_low"]].index)
        elif vital == "MAP":
            idxs += list(series[series > TH["MAP_high"]].index)
            idxs += list(series[series < TH["MAP_low"]].index)
        elif vital == "RR":
            idxs += list(series[series > TH["RR_tachypnea"]].index)
            idxs += list(series[series < TH["RR_apnea"]].index)
        elif vital == "SpO2":
            idxs += list(series[series < TH["SpO2_low"]].index)
        elif vital == "Temp":
            idxs += list(series[series > TH["Temp_high"]].index)
            idxs += list(series[series < TH["Temp_low"]].index)
    else:
        z = (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) != 0 else 1)
        idxs += list(series[np.abs(z) > 3].index)
    return sorted(list(set(idxs)))

def correct_with_autoencoder(df_numeric, epochs=30, batch_size=32, verbose=0):
    X = df_numeric.values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]
    
    if TF_AVAILABLE:
        ae = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(max(16, input_dim*2), activation="relu"),
            layers.Dense(max(8, input_dim), activation="relu"),
            layers.Dense(max(16, input_dim*2), activation="relu"),
            layers.Dense(input_dim, activation="linear")
        ])
        ae.compile(optimizer="adam", loss="mse")
        ae.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size,
               validation_split=0.1, verbose=verbose)
        recon = ae.predict(X_scaled)
        df_corrected = pd.DataFrame(scaler.inverse_transform(recon),
                                    columns=df_numeric.columns, index=df_numeric.index)
        mse_per_row = np.mean((X_scaled - recon)**2, axis=1)
        return df_corrected, mse_per_row
    else:
        print("TensorFlow not available; skipping autoencoder correction")
        return df_numeric, np.zeros(X.shape[0])

def plot_anomalies(df_original, df_corrected, anomalies, file_name):
    plt.figure(figsize=(14,6))
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.plot(df_original[col], label=f"{col} original", alpha=0.5)
        plt.plot(df_corrected[col], label=f"{col} corrected", alpha=0.8)
    for anom in anomalies:
        if anom["vital"] in df_original.columns:
            plt.scatter(anom["index"], df_original.loc[anom["index"], anom["vital"]],
                        color="red", marker="x")
    plt.title(f"Anomaly Correction - {file_name}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file_name}_anomaly_plot.png")
    plt.close()

# ----------------------
# File Processing
# ----------------------
def process_file(filepath):
    print(f"\n--- Processing {filepath.name} ---")
    df = pd.read_csv(filepath, low_memory=False)
    
    # Convert all to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(axis=1, how="all")
    if df.empty:
        print("No numeric data; skipping.")
        return
    
    anomalies = []
    df_clean = df.copy()
    for col in df_clean.columns:
        idxs = detect_anomalies(df_clean[col], col)
        for i in idxs:
            anomalies.append({"vital": col, "index": int(i)})
            df_clean.loc[i, col] = np.nan
    
    df_clean = df_clean.interpolate().ffill().bfill()
    df_clean = df_clean.apply(pd.to_numeric, errors="coerce")
    
    df_numeric = df_clean.select_dtypes(include=[np.number])
    df_corrected, mse_per_row = correct_with_autoencoder(df_numeric)
    
    # Only store plot
    plot_anomalies(df, df_corrected, anomalies, filepath.stem)
    
    # Print reconstruction accuracy
    mse = mse_per_row.mean()
    r2 = r2_score(df_numeric, df_corrected)
    approx_acc = 100 * (1 - mse)
    print(f"Reconstruction MSE: {mse:.4f}, R²: {r2:.4f}, Approx Accuracy: {approx_acc:.2f}%")

# ----------------------
# Run All Files
# ----------------------
def run_all():
    csv_files = list(DATASETS_DIR.rglob("*.csv"))
    for f in csv_files:
        try:
            process_file(f)
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

if __name__ == "__main__":
    print("ICU Anomaly Pipeline - Model2Results")
    print("TensorFlow available:", TF_AVAILABLE)
    run_all()