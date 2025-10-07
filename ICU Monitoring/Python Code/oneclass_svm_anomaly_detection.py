import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ---------- Config ----------
DATASETS_DIR = Path(r"d:\Anomaly Detection\ICU Monitoring\Datasets")
OUTPUT_DIR = DATASETS_DIR / "../OneClassSVM-ModelResult"
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

# ---------- Functions ----------
def detect_anomalies(series, col_name):
    anomalies = []
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if series.empty:
        return anomalies
    vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
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

def plot_corrected(df_original, df_corrected, anomalies, file_name, scores=None, threshold=None, score_label="Score"):
    plt.figure(figsize=(14,6))
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_corrected.columns:
            plt.plot(df_original[col], label=f"{col} original", alpha=0.5)
            plt.plot(df_corrected[col], label=f"{col} corrected", alpha=0.9)
    for anom in anomalies:
        if anom["vital"] in df_corrected.columns:
            plt.scatter(anom["index"], df_original.loc[anom["index"], anom["vital"]], color="red", marker="x")
    if scores is not None:
        plt.twinx().plot(scores, alpha=0.2, label=score_label)
        if threshold is not None:
            plt.twinx().axhline(y=threshold, color='r', linestyle='--')
    plt.title(f"One-Class SVM Correction - {file_name}")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file_name}_ocsvm_correction.png")
    plt.close()

# ---------- Main Loop ----------
accuracy_summary = []

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} (One-Class SVM) ===")
    try:
        df = pd.read_csv(file, low_memory=False)
        df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        if df.empty:
            print("No numeric cols.")
            continue

        # Threshold anomalies
        threshold_anoms = []
        for col in df.columns:
            threshold_anoms += detect_anomalies(df[col], col)

        df_clean = df.copy()
        for a in threshold_anoms:
            if a['vital'] in df_clean.columns:
                df_clean.loc[a['index'], a['vital']] = np.nan
        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.loc[:, df_clean.nunique() > 1].fillna(df_clean.mean())

        # Scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean.values.astype(float))

        # One-Class SVM
        oc = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
        oc.fit(X)
        scores = oc.decision_function(X)
        thresh = np.percentile(scores, 5)
        anomalies = scores < thresh
        anomaly_count = anomalies.sum()

        # Correct anomalous rows
        df_corrected = df_clean.copy()
        df_corrected.loc[anomalies, :] = np.nan
        df_corrected = df_corrected.interpolate().ffill().bfill()

        # Save corrected CSV and anomaly results
        corrected_path = OUTPUT_DIR / f"{file.stem}_corrected_oneclasssvm.csv"
        df_corrected.to_csv(corrected_path, index=False)
        results = pd.DataFrame({"DecisionScore": scores, "Anomaly": anomalies.astype(int)})
        results.to_csv(OUTPUT_DIR / f"{file.stem}_oneclasssvm_anomalies.csv", index=False)

        # Plot
        plot_corrected(df, df_corrected, threshold_anoms, file.stem, scores=-scores, threshold=-thresh, score_label="(-)DecisionScore")

        # Compute metrics (compare corrected vs clean)
        mse = np.mean((df_clean.values - df_corrected.values)**2)
        r2 = 1 - mse / np.var(df_clean.values)
        acc = 100 * (1 - mse / np.var(df_clean.values))
        print(f"File Accuracy≈{acc:.2f}% | Anomalies={anomaly_count}")

        # Save for summary
        accuracy_summary.append({
            "File": file.name,
            "MSE": mse,
            "R2": r2,
            "Accuracy (%)": acc,
            "Anomalies Detected": anomaly_count
        })

    except Exception as e:
        print(f"Error {file.name}: {e}")

# ---------- Overall Accuracy ----------
if accuracy_summary:
    summary_df = pd.DataFrame(accuracy_summary)
    avg_acc = summary_df["Accuracy (%)"].mean()
    summary_df["Average Accuracy (%)"] = avg_acc

    summary_path = OUTPUT_DIR / "oneclasssvm_accuracy_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ One-Class SVM Accuracy Summary saved → {summary_path}")
    print(summary_df)
    print(f"\n📊 Overall Average One-Class SVM Accuracy: {avg_acc:.2f}%")
else:
    print("⚠️ No results to summarize.")
