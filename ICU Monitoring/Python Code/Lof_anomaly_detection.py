import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# Paths & Config
# ======================
DATASETS_DIR = Path(r"d:\Anomaly Detection\ICU Monitoring\Datasets")
OUTPUT_DIR = DATASETS_DIR / "../LOF-ModelResults"
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
# Anomaly detection (threshold-based)
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
        z = (series - series.mean()) / series.std(ddof=0)
        anomalies += [{"vital": col_name, "index": i} for i in series[np.abs(z) > 3].index]

    return anomalies

# ======================
# Plot anomalies
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
    plt.title(f"LOF Anomaly Correction - {file_name}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file_name}_LOF_corrected_graph.png")
    plt.close()

# ======================
# Process all files
# ======================
accuracy_list = []

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} ===")
    try:
        df = pd.read_csv(file, low_memory=False)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(axis=1, how='all')
        if df.empty:
            print("⚠️ No valid numeric data found.")
            continue

        anomalies = []
        for col in df.columns:
            anomalies += detect_anomalies(df[col], col)

        df_clean = df.copy()
        for anom in anomalies:
            df_clean.loc[anom['index'], anom['vital']] = np.nan

        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
        df_clean = df_clean.loc[:, df_clean.nunique() > 1]
        df_clean = df_clean.fillna(df_clean.mean())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean.values.astype(float))

        # LOF Model
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        y_pred = lof.fit_predict(X_scaled)

        df_corrected = df_clean.copy()
        anomalies_idx = np.where(y_pred == -1)[0]
        for i in anomalies_idx:
            df_corrected.iloc[i] = np.nan
        df_corrected = df_corrected.interpolate().ffill().bfill()

        # Save corrected CSV
        corrected_path = OUTPUT_DIR / f"{file.stem}_LOF_corrected.csv"
        df_corrected.to_csv(corrected_path, index=False)
        print(f"✅ LOF corrected data saved → {corrected_path}")
        print(f"Detected {len(anomalies_idx)} anomalies using LOF.")

        # Plot before/after
        plot_corrected(df, df_corrected, anomalies, file.stem)

        # Accuracy Metrics
       # Accuracy Metrics (updated)
        row_errors = np.mean((df_clean - df_corrected)**2, axis=1)
        mse = row_errors.mean()
        r2 = r2_score(df_clean, df_corrected)
        accuracy = 100 * (1 - mse / row_errors.max())  # normalized
        accuracy_list.append(accuracy)
        print(f"LOF Model → MSE: {mse:.4f}, R²: {r2:.4f}, Approx. Accuracy: {accuracy:.2f}%")


        # Save anomaly results
        errors_df = pd.DataFrame({
            "Index": np.arange(len(df_clean)),
            "Anomaly": (y_pred == -1).astype(int)
        })
        errors_df.to_csv(OUTPUT_DIR / f"{file.stem}_LOF_anomaly_results.csv", index=False)
        print(f"Anomaly results saved → {file.stem}_LOF_anomaly_results.csv")

    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")

# Overall accuracy across all datasets
if accuracy_list:
    overall_accuracy = np.mean(accuracy_list)
    print(f"\n📊 Overall LOF Accuracy across all datasets: {overall_accuracy:.2f}%")
