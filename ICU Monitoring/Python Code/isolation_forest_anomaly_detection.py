import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# Directories
# ======================
DATASETS_DIR = Path(r"d:\Anomaly Detection\ICU Monitoring\Datasets")
OUTPUT_DIR = DATASETS_DIR / "../Isolation-ModelResults"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Thresholds and Vital Sign Mappings
# ======================
TH = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90,
    "DBP_high": 90, "DBP_low": 60,
    "MAP_high": 110, "MAP_low": 70,
    "RR_tachypnea": 24, "RR_apnea": 8,
    "SpO2_low": 90, "Temp_high": 38.5, "Temp_low": 35.0
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

# ======================
# Detect anomalies based on thresholds
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
        # generic z-score detection
        z = (series - series.mean()) / series.std(ddof=0)
        anomalies += [{"vital": col_name, "index": i} for i in series[np.abs(z) > 3].index]

    return anomalies

# ======================
# Plot corrected vs original
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
    plt.title(f"Anomaly Correction (Isolation Forest) - {file_name}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file_name}_isoforest_plot.png")
    plt.close()

# ======================
# Main processing loop
# ======================
accuracy_summary = []

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} (Isolation Forest) ===")
    try:
        df = pd.read_csv(file, low_memory=False)
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        if df.empty:
            print("⚠️ Skipped empty dataset.")
            continue

        # Detect anomalies
        anomalies = []
        for col in df.columns:
            anomalies += detect_anomalies(df[col], col)

        df_clean = df.copy()
        for a in anomalies:
            if a['vital'] in df_clean.columns:
                df_clean.loc[a['index'], a['vital']] = np.nan
        df_clean = df_clean.interpolate().ffill().bfill()

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean)

        # Isolation Forest
        model = IsolationForest(contamination=0.05, random_state=42)
        preds = model.fit_predict(X_scaled)

        # Mark anomalies
        anomaly_mask = preds == -1
        df_corrected = df_clean.copy()
        df_corrected.loc[anomaly_mask] = np.nan
        df_corrected = df_corrected.interpolate().ffill().bfill()

        # Save corrected data
        out_path = OUTPUT_DIR / f"{file.stem}_corrected_isoforest.csv"
        df_corrected.to_csv(out_path, index=False)
        print(f"✅ Corrected data saved → {out_path}")

        # Plot graph
        plot_corrected(df, df_corrected, anomalies, file.stem)

        # Compute metrics
        mse = mean_squared_error(df_clean, df_corrected)
        r2 = r2_score(df_clean, df_corrected)
        acc = 100 * (1 - mse / np.var(df_clean.values))

        print(f"MSE={mse:.6f} | R²={r2:.4f} | Accuracy≈{acc:.2f}% | Anomalies={len(anomalies)}")

        accuracy_summary.append({
            "File": file.name,
            "MSE": mse,
            "R2": r2,
            "Accuracy (%)": acc,
            "Anomalies Detected": len(anomalies)
        })

    except Exception as e:
        print(f"❌ Error in {file.name}: {e}")

# ======================
# Save overall accuracy summary
# ======================
if accuracy_summary:
    summary_df = pd.DataFrame(accuracy_summary)
    avg_acc = summary_df["Accuracy (%)"].mean()
    summary_df["Average Accuracy (%)"] = avg_acc

    summary_path = OUTPUT_DIR / "isolationforest_accuracy_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Isolation Forest Accuracy Summary saved → {summary_path}")
    print(summary_df)
    print(f"\n📊 Overall Average Isolation Forest Accuracy: {avg_acc:.2f}%")
else:
    print("\n⚠️ No results generated to summarize.")
