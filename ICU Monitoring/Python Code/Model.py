import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# Paths & Config
# ======================
DATASETS_DIR = Path(r"d:\Anomaly Detection\ICU Monitoring\Datasets")
OUTPUT_DIR = DATASETS_DIR / "../ModelResults"
OUTPUT_DIR.mkdir(exist_ok=True)

# ======================
# Thresholds for vitals
# ======================
TH = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90,
    "DBP_high": 90, "DBP_low": 60,
    "MAP_high": 110, "MAP_low": 70,
    "RR_tachypnea": 24, "RR_apnea": 8,
    "SpO2_low": 90,
    "Temp_high": 38.5, "Temp_low": 35.0
}

# Vital column mapping
VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR", "PR"],
    "SBP": ["SysBP", "SBP", "SystolicBP", "SYS"],
    "DBP": ["DiaBP", "DBP", "DiastolicBP", "DIA"],
    "MAP": ["MAP", "MeanBP", "MeanArterialPressure"],
    "RR": ["RespRate", "RR", "Resp", "RespiratoryRate"],
    "SpO2": ["SpO2", "OxygenSaturation", "O2Sat", "SaO2"],
    "Temp": ["Temp", "Temperature", "BodyTemp", "T"]
}

# ======================
# Anomaly detection
# ======================
def detect_anomalies(series, col_name):
    anomalies = []
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if series.empty:
        return anomalies

    vital = None
    for v, cols in VITAL_MAPPING.items():
        if col_name in cols:
            vital = v
            break

    if vital:
        if vital == "HR":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > TH["HR_tachy"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < TH["HR_brady"]].index]
        elif vital == "SBP":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > TH["SBP_high"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < TH["SBP_low"]].index]
        elif vital == "DBP":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > TH["DBP_high"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < TH["DBP_low"]].index]
        elif vital == "MAP":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > TH["MAP_high"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < TH["MAP_low"]].index]
        elif vital == "RR":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > TH["RR_tachypnea"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < TH["RR_apnea"]].index]
        elif vital == "SpO2":
            anomalies += [{"vital": col_name, "index": i} for i in series[series < TH["SpO2_low"]].index]
        elif vital == "Temp":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > TH["Temp_high"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < TH["Temp_low"]].index]
    else:
        # Generic z-score
        z = (series - series.mean()) / series.std(ddof=0)
        anomalies += [{"vital": col_name, "index": i} for i in series[np.abs(z) > 3].index]

    return anomalies

# ======================
# Plot before/after anomalies
# ======================
def plot_corrected(df_original, df_corrected, anomalies, file_name):
    plt.figure(figsize=(14,6))
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_corrected.columns:
            plt.plot(df_original[col], label=f"{col} original", alpha=0.5)
            plt.plot(df_corrected[col], label=f"{col} corrected", alpha=0.9)
    for anom in anomalies:
        if anom["vital"] in df_corrected.columns:
            plt.scatter(anom["index"], df_original.loc[anom["index"], anom["vital"]], color="red", marker="x")
    plt.title(f"Anomaly Correction - {file_name}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file_name}_corrected_graph.png")
    plt.close()

# ======================
# Process all files
# ======================
for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} ===")
    try:
        df = pd.read_csv(file, low_memory=False)

        # Convert all columns to numeric if possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop columns with all NaNs
        df = df.dropna(axis=1, how='all')
        if df.empty:
            print("⚠️ No valid numeric data found.")
            continue

        # Detect anomalies
        anomalies = []
        for col in df.columns:
            anomalies += detect_anomalies(df[col], col)

        # Mask anomalies
        df_clean = df.copy()
        for anom in anomalies:
            df_clean.loc[anom['index'], anom['vital']] = np.nan

        # Fill missing values
        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.apply(pd.to_numeric, errors='coerce')

        # Drop constant columns (zero variance)
        df_clean = df_clean.loc[:, df_clean.nunique() > 1]

        # Safety check: replace remaining NaNs with column mean
        df_clean = df_clean.fillna(df_clean.mean())

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean.values.astype(float))  # ensure float

        # Build Autoencoder
        input_dim = X_scaled.shape[1]
        autoencoder = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

        # Correct data using Autoencoder
        reconstructions = autoencoder.predict(X_scaled)
        df_corrected = pd.DataFrame(scaler.inverse_transform(reconstructions), columns=df_clean.columns)

        # Save corrected CSV
        corrected_path = OUTPUT_DIR / f"{file.stem}_corrected.csv"
        df_corrected.to_csv(corrected_path, index=False)
        print(f"✅ Corrected data saved → {corrected_path}")
        print(f"Detected {len(anomalies)} anomalies, corrected using Autoencoder.")

        # Plot before/after
        plot_corrected(df, df_corrected, anomalies, file.stem)

    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")

print()

history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
print(f"Final training loss: {history.history['loss'][-1]:.4f}")
print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

reconstructions = autoencoder.predict(X_scaled)

# Compute reconstruction error (MSE)
mse = mean_squared_error(X_scaled, reconstructions)
print(f"Reconstruction MSE: {mse:.4f}")

# Optionally compute R² score (like a regression accuracy)
r2 = r2_score(X_scaled, reconstructions)
print(f"Reconstruction R² Score: {r2:.4f}")

# You can also compute reconstruction "accuracy" as similarity %
accuracy = 100 * (1 - mse)
print(f"Reconstruction Accuracy (approx): {accuracy:.2f}%")

mse_per_row = np.mean(np.square(X_scaled - reconstructions), axis=1)

# Put errors in a DataFrame
errors_df = pd.DataFrame({
    "ReconstructionError": mse_per_row
})

# Set anomaly threshold (95th percentile is common)
threshold = np.percentile(mse_per_row, 95)
print(f"Anomaly Detection Threshold (95th percentile): {threshold:.6f}")

# Label anomalies
errors_df["Anomaly"] = errors_df["ReconstructionError"] > threshold

# Count anomalies
num_anomalies = errors_df["Anomaly"].sum()
print(f"Detected {num_anomalies} anomalies out of {len(errors_df)} rows")

# Optional: Save anomaly results
errors_df.to_csv(OUTPUT_DIR / "../anomaly_detection_results.csv", index=False)
print("Anomaly detection results saved → anomaly_detection_results.csv")