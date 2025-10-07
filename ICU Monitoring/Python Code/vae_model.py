# vae_model_fixed_layer.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, backend as K

# ---------- Config ----------
DATASETS_DIR = Path(r"d:\Anomaly Detection\ICU Monitoring\Datasets")
OUTPUT_DIR = DATASETS_DIR / "../VAE-ModelResult"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Thresholds & Vital mapping ----------
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

# ---------- Sampling Layer ----------
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

# ---------- Main ----------
accuracy_list = []

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} (VAE) ===")
    try:
        df = pd.read_csv(file, low_memory=False)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(axis=1, how='all')
        if df.empty:
            continue

        # Threshold anomalies
        threshold_anoms = []
        for c in df.columns:
            threshold_anoms += detect_anomalies(df[c], c)

        df_clean = df.copy()
        for a in threshold_anoms:
            if a['vital'] in df_clean.columns and a['index'] < len(df_clean):
                df_clean.loc[a['index'], a['vital']] = np.nan

        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.loc[:, df_clean.nunique() > 1].fillna(df_clean.mean())

        # Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean.values.astype(float))
        n_features = X.shape[1]
        latent_dim = min(16, max(2, n_features // 2))

        # ---------- Encoder ----------
        inputs = layers.Input(shape=(n_features,))
        h = layers.Dense(64, activation='relu')(inputs)
        h = layers.Dense(32, activation='relu')(h)
        z_mean = layers.Dense(latent_dim)(h)
        z_log_var = layers.Dense(latent_dim)(h)
        z = Sampling()([z_mean, z_log_var])

        # ---------- Decoder ----------
        h_dec = layers.Dense(32, activation='relu')(z)
        h_dec = layers.Dense(64, activation='relu')(h_dec)
        outputs = layers.Dense(n_features, activation='linear')(h_dec)

        vae = models.Model(inputs, outputs)

        # ---------- Loss ----------
        def vae_loss(x_true, x_pred):
            recon_loss = K.mean(K.square(x_true - x_pred), axis=-1)
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return recon_loss + 0.001 * kl_loss

        vae.compile(optimizer='adam', loss=vae_loss)
        vae.fit(X, X, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

        # ---------- Reconstruction ----------
        recon = vae.predict(X)
        recon_inv = scaler.inverse_transform(recon)
        df_corrected = pd.DataFrame(recon_inv, columns=df_clean.columns)
        df_corrected.to_csv(OUTPUT_DIR / f"{file.stem}_corrected_vae.csv", index=False)

        # ---------- Metrics ----------
        mse_per_row = np.mean(np.square(X - recon), axis=1)
        threshold = np.percentile(mse_per_row, 95)
        anomalies = mse_per_row > threshold

        overall_mse = mse_per_row.mean()
        accuracy = 100 * (1 - overall_mse / np.var(X))
        accuracy_list.append(accuracy)

        results = pd.DataFrame({"ReconstructionError": mse_per_row, "Anomaly": anomalies.astype(int)})
        results.to_csv(OUTPUT_DIR / f"{file.stem}_vae_anomalies.csv", index=False)

        # ---------- Plot ----------
        plt.figure(figsize=(10,5))
        plt.plot(mse_per_row, label="Reconstruction Error")
        plt.axhline(y=threshold, color='r', linestyle='--', label='95th percentile')
        plt.title(f"VAE Reconstruction Error - {file.stem}")
        plt.xlabel("Sample Index")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{file.stem}_vae_error_plot.png")
        plt.close()

        print(f"{file.name} → Accuracy: {accuracy:.2f}% | MSE: {overall_mse:.6f} | Anomalies: {anomalies.sum()}")

    except Exception as e:
        print(f"Error {file.name}: {e}")

if accuracy_list:
    print(f"\n📊 Overall VAE Accuracy: {np.mean(accuracy_list):.2f}%")
