# lstm_autoencoder_model.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------- config ----------
DATASETS_DIR = Path(r"d:\Anomaly Detection\ICU Monitoring\Datasets")
OUTPUT_DIR = DATASETS_DIR / "../LSTM_AE-ModelResult"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TH = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90,
    "DBP_high": 90, "DBP_low": 60,
    "MAP_high": 110, "MAP_low": 70,
    "RR_tachypnea": 24, "RR_apnea": 8,
    "SpO2_low": 90, "Temp_high": 38.5, "Temp_low": 35.0
}

VITAL_MAPPING = {
    "HR": ["Pulse","HeartRate","HR","PR"],
    "SBP": ["SysBP","SBP","SystolicBP","SYS"],
    "DBP": ["DiaBP","DBP","DiastolicBP","DIA"],
    "MAP": ["MAP","MeanBP","MeanArterialPressure"],
    "RR": ["RespRate","RR","Resp","RespiratoryRate"],
    "SpO2": ["SpO2","OxygenSaturation","O2Sat","SaO2"],
    "Temp": ["Temp","Temperature","BodyTemp","T"]
}

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

def create_sequences(X, seq_len=10):
    n_samples, n_feats = X.shape
    seqs = []
    for i in range(n_samples - seq_len + 1):
        seqs.append(X[i:i+seq_len])
    return np.array(seqs)

def reconstruct_sequences_to_rows(seq_reconstructions, original_length, seq_len=10):
    n_seq, _, n_feats = seq_reconstructions.shape
    accum = np.zeros((original_length, n_feats))
    counts = np.zeros((original_length, 1))
    for i in range(n_seq):
        start = i
        end = i + seq_len
        accum[start:end] += seq_reconstructions[i]
        counts[start:end] += 1
    counts[counts == 0] = 1
    return accum / counts

# ---------- main ----------
SEQ_LEN = 10
accuracy_list = []

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} (LSTM Autoencoder) ===")
    try:
        df = pd.read_csv(file, low_memory=False)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(axis=1, how='all')
        if df.empty:
            print("No numeric columns.")
            continue

        threshold_anoms = []
        for c in df.columns:
            threshold_anoms += detect_anomalies(df[c], c)

        df_clean = df.copy()
        for a in threshold_anoms:
            if a['vital'] in df_clean.columns and a['index'] < len(df_clean):
                df_clean.loc[a['index'], a['vital']] = np.nan

        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.loc[:, df_clean.nunique() > 1].fillna(df_clean.mean())

        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean.values.astype(float))
        if len(X) < SEQ_LEN:
            print("Not enough rows for sequence modelling — skipping.")
            continue

        seqs = create_sequences(X, seq_len=SEQ_LEN)
        n_feats = seqs.shape[2]

        # LSTM Autoencoder
        inp = layers.Input(shape=(SEQ_LEN, n_feats))
        x = layers.LSTM(64, return_sequences=True)(inp)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.RepeatVector(SEQ_LEN)(x)
        x = layers.LSTM(32, return_sequences=True)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        out = layers.TimeDistributed(layers.Dense(n_feats))(x)
        model = models.Model(inp, out)
        model.compile(optimizer='adam', loss='mse')
        model.fit(seqs, seqs, epochs=30, batch_size=32, validation_split=0.1, verbose=0)

        seq_recon = model.predict(seqs)
        reconstructed_rows = reconstruct_sequences_to_rows(seq_recon, original_length=X.shape[0], seq_len=SEQ_LEN)
        recon_inv = scaler.inverse_transform(reconstructed_rows)
        df_corrected = pd.DataFrame(recon_inv, columns=df_clean.columns)
        corrected_path = OUTPUT_DIR / f"{file.stem}_corrected_lstm_ae.csv"
        df_corrected.to_csv(corrected_path, index=False)

        # MSE and overall accuracy
        mse_per_row = np.mean(np.square(X - reconstructed_rows), axis=1)
        overall_mse = mse_per_row.mean()
        accuracy = 100 * (1 - overall_mse / np.var(X))
        accuracy_list.append(accuracy)

        # Anomaly detection
        threshold = np.percentile(mse_per_row, 95)
        anomalies = mse_per_row > threshold
        results = pd.DataFrame({"ReconstructionError": mse_per_row, "Anomaly": anomalies.astype(int)})
        results.to_csv(OUTPUT_DIR / f"{file.stem}_lstm_ae_anomalies.csv", index=False)

        # Plot reconstruction error
        plt.figure(figsize=(10,5))
        plt.plot(mse_per_row, label="Reconstruction Error")
        plt.axhline(y=threshold, color='r', linestyle='--', label='95th percentile')
        plt.title(f"LSTM-AE Reconstruction Error - {file.stem}")
        plt.xlabel("Sample Index")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{file.stem}_lstm_ae_error_plot.png")
        plt.close()

        print(f"{file.name} → Overall Accuracy: {accuracy:.2f}% | Avg MSE: {overall_mse:.6f} | Anomalies: {anomalies.sum()}/{len(anomalies)}")

    except Exception as e:
        print(f"Error {file.name}: {e}")

# Overall average accuracy
if accuracy_list:
    overall_avg_accuracy = np.mean(accuracy_list)
    print(f"\n📊 Overall LSTM-Autoencoder Accuracy across all datasets: {overall_avg_accuracy:.2f}%")
