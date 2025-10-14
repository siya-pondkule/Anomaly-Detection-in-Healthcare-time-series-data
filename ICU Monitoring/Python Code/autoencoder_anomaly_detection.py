#Learn the normal physiological patterns of ICU vital signs,
#and then detect anomalies (unusual readings) based on how poorly they can be reconstructed.

#The model learns “what normal looks like.”
#Anything that doesn’t look normal produces high reconstruction error and flagged as anomaly.

# working of model

#The encoder learns to compress the data (understand normal patterns).
#The decoder learns to recreate the data from that compressed form.
#When something is unusual or abnormal, the model can’t rebuild it accurately, so the difference (called reconstruction error) becomes large — that’s how an anomaly is detected.


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# Paths
# ======================
DATASETS_DIR = Path(r"d:\Anomaly Detection\ICU Monitoring\Datasets")
OUTPUT_DIR = DATASETS_DIR / "../Autoencode-ModelResults"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Thresholds for vital signs
# ======================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90,
    "DBP_high": 90, "DBP_low": 60,
    "MAP_high": 110, "MAP_low": 70,
    "RR_tachypnea": 24, "RR_apnea": 8,
    "SpO2_low": 90,
    "Temp_high": 38.5, "Temp_low": 35.0
}

# Vital column name mapping
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
# Detect anomalies in vital series
# ======================
def detect_anomalies(series, col_name):
    anomalies = []
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if series.empty:
        return anomalies

    vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)

    if vital:
        if vital == "HR":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > THRESHOLDS["HR_tachy"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < THRESHOLDS["HR_brady"]].index]
        elif vital == "SBP":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > THRESHOLDS["SBP_high"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < THRESHOLDS["SBP_low"]].index]
        elif vital == "DBP":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > THRESHOLDS["DBP_high"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < THRESHOLDS["DBP_low"]].index]
        elif vital == "MAP":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > THRESHOLDS["MAP_high"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < THRESHOLDS["MAP_low"]].index]
        elif vital == "RR":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > THRESHOLDS["RR_tachypnea"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < THRESHOLDS["RR_apnea"]].index]
        elif vital == "SpO2":
            anomalies += [{"vital": col_name, "index": i} for i in series[series < THRESHOLDS["SpO2_low"]].index]
        elif vital == "Temp":
            anomalies += [{"vital": col_name, "index": i} for i in series[series > THRESHOLDS["Temp_high"]].index]
            anomalies += [{"vital": col_name, "index": i} for i in series[series < THRESHOLDS["Temp_low"]].index]
    else:
        z = (series - series.mean()) / series.std(ddof=0)
        anomalies += [{"vital": col_name, "index": i} for i in series[np.abs(z) > 3].index]

    return anomalies


# ======================
# Plot corrected vs original data
# ======================
def plot_corrected(df_original, df_corrected, anomalies, file_name):
    plt.figure(figsize=(14,6))
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_corrected.columns:
            plt.plot(df_original[col], label=f"{col} (original)", alpha=0.5)
            plt.plot(df_corrected[col], label=f"{col} (corrected)", alpha=0.9)
    for anom in anomalies:
        if anom["vital"] in df_corrected.columns:
            plt.scatter(anom["index"], df_original.loc[anom["index"], anom["vital"]], color="red", marker="x")
    plt.title(f"Anomaly Correction - {file_name}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file_name}_correction_plot.png")
    plt.close()


# ======================
# Main Processing Loop
# ======================
accuracy_summary = []

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} ===")
    try:
        df = pd.read_csv(file, low_memory=False)

        # Convert all to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(axis=1, how='all')

        if df.empty:
            print("⚠️ No valid numeric data found.")
            continue

        # Detect anomalies by vital
        anomalies = []
        for col in df.columns:
            anomalies += detect_anomalies(df[col], col)

        # Replace anomalies with NaN
        df_clean = df.copy()
        for anom in anomalies:
            if anom["vital"] in df_clean.columns:
                df_clean.loc[anom["index"], anom["vital"]] = np.nan

        # Fill missing data
        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.loc[:, df_clean.nunique() > 1].fillna(df_clean.mean())

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean.values.astype(float))

        # Autoencoder Model
        input_dim = X_scaled.shape[1]
        autoencoder = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])
        autoencoder.compile(optimizer='adam', loss='mse')

        # Train autoencoder
        history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
        print(f"Final training loss: {history.history['loss'][-1]:.4f}, val_loss: {history.history['val_loss'][-1]:.4f}")

        # Reconstruct (correct) data
        reconstructions = autoencoder.predict(X_scaled)
        df_corrected = pd.DataFrame(scaler.inverse_transform(reconstructions), columns=df_clean.columns)

        # Save corrected file
        corrected_path = OUTPUT_DIR / f"{file.stem}_corrected.csv"
        df_corrected.to_csv(corrected_path, index=False)
        print(f"✅ Corrected data saved → {corrected_path}")

        # Plot before vs after
        plot_corrected(df, df_corrected, anomalies, file.stem)

        # Evaluate reconstruction
        mse = mean_squared_error(X_scaled, reconstructions)
        r2 = r2_score(X_scaled, reconstructions)
        accuracy = 100 * (1 - mse)
        print(f"MSE={mse:.4f} | R²={r2:.4f} | Accuracy≈{accuracy:.2f}% | Anomalies={len(anomalies)}")

        # Save anomaly scores
        mse_per_row = np.mean(np.square(X_scaled - reconstructions), axis=1)
        threshold = np.percentile(mse_per_row, 95)
        anomalies_mask = mse_per_row > threshold

        results_df = pd.DataFrame({
            "ReconstructionError": mse_per_row,
            "Anomaly": anomalies_mask
        })
        results_df.to_csv(OUTPUT_DIR / f"{file.stem}_anomaly_results.csv", index=False)

        # Plot reconstruction error
        plt.figure(figsize=(10,5))
        plt.plot(mse_per_row, label="Reconstruction Error")
        plt.axhline(y=threshold, color='r', linestyle='--', label='95th percentile threshold')
        plt.title(f"Reconstruction Error Plot - {file.stem}")
        plt.xlabel("Sample Index")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{file.stem}_error_plot.png")
        plt.close()

        # Add to summary
        accuracy_summary.append({
            "File": file.name,
            "MSE": mse,
            "R2": r2,
            "Accuracy (%)": accuracy,
            "Anomalies Detected": len(anomalies)
        })

    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")

# ======================
# Save Overall Summary
# ======================
if accuracy_summary:
    summary_df = pd.DataFrame(accuracy_summary)
    summary_df["Average Accuracy (%)"] = summary_df["Accuracy (%)"].mean()
    summary_path = OUTPUT_DIR / "autoencoder_accuracy_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Overall Model Accuracy Summary saved → {summary_path}")
    print(summary_df)
    print(f"\n📊 Overall Average Autoencoder Accuracy: {summary_df['Accuracy (%)'].mean():.2f}%")
else:
    print("\n⚠️ No valid results to summarize.")
