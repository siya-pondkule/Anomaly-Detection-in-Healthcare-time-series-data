import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# Paths & Config
# ======================
DATASETS_DIR = Path(r"d:\Anomaly Detection\ICU Monitoring\Datasets")
OUTPUT_DIR = DATASETS_DIR / "../AnomalyResults"
OUTPUT_DIR.mkdir(exist_ok=True)

# ======================
# Thresholds (existing vitals)
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

# ======================
# Vital Mapping
# ======================
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
# Anomaly Detection Function
# ======================
def detect_anomalies_generic(series, col_name, anomalies):
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if series.empty:
        return

    # Check if column matches known vitals
    vital = None
    for v, cols in VITAL_MAPPING.items():
        if col_name in cols:
            vital = v
            break

    if vital:
        if vital == "HR":
            anomalies += [{"vital": col_name, "anomaly": "tachycardia", "index": i, "value": series.loc[i]}
                          for i in series[series > TH["HR_tachy"]].index]
            anomalies += [{"vital": col_name, "anomaly": "bradycardia", "index": i, "value": series.loc[i]}
                          for i in series[series < TH["HR_brady"]].index]

        elif vital == "SBP":
            anomalies += [{"vital": col_name, "anomaly": "hypertensive_spike", "index": i, "value": series.loc[i]}
                          for i in series[series > TH["SBP_high"]].index]
            anomalies += [{"vital": col_name, "anomaly": "hypotension", "index": i, "value": series.loc[i]}
                          for i in series[series < TH["SBP_low"]].index]

        elif vital == "DBP":
            anomalies += [{"vital": col_name, "anomaly": "high_diastolic", "index": i, "value": series.loc[i]}
                          for i in series[series > TH["DBP_high"]].index]
            anomalies += [{"vital": col_name, "anomaly": "low_diastolic", "index": i, "value": series.loc[i]}
                          for i in series[series < TH["DBP_low"]].index]

        elif vital == "MAP":
            anomalies += [{"vital": col_name, "anomaly": "high_map", "index": i, "value": series.loc[i]}
                          for i in series[series > TH["MAP_high"]].index]
            anomalies += [{"vital": col_name, "anomaly": "low_map", "index": i, "value": series.loc[i]}
                          for i in series[series < TH["MAP_low"]].index]

        elif vital == "RR":
            anomalies += [{"vital": col_name, "anomaly": "tachypnea", "index": i, "value": series.loc[i]}
                          for i in series[series > TH["RR_tachypnea"]].index]
            anomalies += [{"vital": col_name, "anomaly": "apnea_event", "index": i, "value": series.loc[i]}
                          for i in series[series < TH["RR_apnea"]].index]

        elif vital == "SpO2":
            anomalies += [{"vital": col_name, "anomaly": "hypoxemia", "index": i, "value": series.loc[i]}
                          for i in series[series < TH["SpO2_low"]].index]

        elif vital == "Temp":
            anomalies += [{"vital": col_name, "anomaly": "hyperpyrexia", "index": i, "value": series.loc[i]}
                          for i in series[series > TH["Temp_high"]].index]
            anomalies += [{"vital": col_name, "anomaly": "hypothermia", "index": i, "value": series.loc[i]}
                          for i in series[series < TH["Temp_low"]].index]
    else:
        # Generic z-score for unknown numeric columns
        z = (series - series.mean()) / series.std(ddof=0)
        anomalies += [{"vital": col_name, "anomaly": "outlier", "index": i, "value": series.loc[i]}
                      for i in series[np.abs(z) > 3].index]

# ======================
# Combined Plot Function
# ======================
def plot_anomalies_combined(df, anomalies, graph_path):
    plt.figure(figsize=(14,6))
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna().reset_index(drop=True)
        if not series.empty:
            plt.plot(series.index, series, label=col)
    for anom in anomalies:
        plt.scatter(anom["index"], anom["value"], color="red", marker="x")
    plt.title("Anomaly Detection - All Columns")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

# ======================
# Process All CSV Files
# ======================
for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} ===")
    try:
        df = pd.read_csv(file, low_memory=False)
        anomalies = []

        # Convert all columns to numeric, ignore errors
        df = df.apply(pd.to_numeric, errors='coerce')

        # Select columns with at least one numeric value
        numeric_cols = df.columns[df.notna().any()].tolist()
        if not numeric_cols:
            print("No numeric columns found.")
            continue

        # Detect anomalies for all numeric columns
        for col in numeric_cols:
            series = df[col].dropna().reset_index(drop=True)
            detect_anomalies_generic(series, col, anomalies)

        # Save anomalies CSV
        if anomalies:
            out_df = pd.DataFrame(anomalies)
            save_path = OUTPUT_DIR / f"{file.stem}_anomalies.csv"
            out_df.to_csv(save_path, index=False)
            print(f"✅ Found {len(anomalies)} anomalies → {save_path}")

            # Plot combined graph
            graph_path = OUTPUT_DIR / f"{file.stem}_anomalies_graph.png"
            plot_anomalies_combined(df[numeric_cols], anomalies, graph_path)
        else:
            print(f"⚠️ No anomalies detected in numeric columns.")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
