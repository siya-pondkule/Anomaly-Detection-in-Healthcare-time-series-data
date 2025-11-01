import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Reshape, Add, Activation, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\TCN-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)
# ======================
# Load and preprocess data
# ======================
print(f"\n=== Processing {DATA_PATH.name} for TCN Autoencoder ===")
df = pd.read_csv(DATA_PATH)

# Select numeric columns only, drop IDs/timestamps
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for TCN anomaly detection.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================
# Create sequences for TCN
# ======================
time_steps = 10  # can tune this
def create_sequences(data, time_steps=10):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

X_seq = create_sequences(X_scaled, time_steps)
print(f"Shape of data for TCN: {X_seq.shape}")  # (samples, timesteps, features)

# ======================
# TCN Block
# ======================
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    prev_x = x
    # Causal convolution
    x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)
    # Residual connection
    if prev_x.shape[-1] != filters:
        prev_x = Conv1D(filters, 1, padding="same")(prev_x)
    x = Add()([x, prev_x])
    x = Activation("relu")(x)
    return x

# ======================
# Build TCN Autoencoder
# ======================
def build_tcn_autoencoder(timesteps, n_features):
    inputs = Input(shape=(timesteps, n_features))

    # Encoder
    x = tcn_block(inputs, 64, 3, dilation_rate=1, dropout_rate=0.1)
    x = tcn_block(x, 64, 3, dilation_rate=2, dropout_rate=0.1)
    encoded = Flatten()(x)

    # Bottleneck
    bottleneck = Dense(32, activation='relu')(encoded)

    # Decoder
    x = Dense(timesteps * 64, activation='relu')(bottleneck)
    x = Reshape((timesteps, 64))(x)
    x = tcn_block(x, 64, 3, dilation_rate=1, dropout_rate=0.1)
    outputs = Conv1D(n_features, 1, activation='linear', padding="same")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_tcn_autoencoder(X_seq.shape[1], X_seq.shape[2])
model.summary()

# ======================
# Train TCN Autoencoder
# ======================
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\nTraining TCN Autoencoder...")
history = model.fit(
    X_seq, X_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stop]
)

# ======================
# Reconstruction and anomaly detection
# ======================
X_pred = model.predict(X_seq)
mse = np.mean(np.mean(np.square(X_seq - X_pred), axis=2), axis=1)  # per sequence MSE

# Use percentile threshold
threshold = np.percentile(mse, 95)
anomalies = np.where(mse > threshold)[0]

print(f"\nDetected {len(anomalies)} anomalies out of {len(mse)} sequences.")
print(f"Anomaly threshold: {threshold:.6f}")

# ======================
# Save results
# ======================
results_df = pd.DataFrame({
    "Sequence_Index": np.arange(len(mse)),
    "Reconstruction_Error": mse,
    "Anomaly_Label": np.where(mse > threshold, "Anomaly", "Normal")
})
results_file = OUTPUT_DIR / "ICUSTAYS_TCN_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# ======================
# Summary
# ======================
summary_data = {
    "Total Sequences": len(mse),
    "Detected Anomalies": len(anomalies),
    "Anomaly Percentage (%)": round(100 * len(anomalies) / len(mse), 2)
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "TCN_ICUSTAYS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved → {summary_file}")

# ======================
# Plot reconstruction error
# ======================
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error", color="blue", alpha=0.7)
plt.scatter(anomalies, mse[anomalies], color="red", label="Detected Anomalies", marker="x")
plt.axhline(threshold, color="orange", linestyle="--", label="Threshold")
plt.title("TCN Autoencoder Anomaly Detection - ICUSTAYS.csv")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_TCN_Anomaly_Plot.png")
plt.close()

print(f"Plot saved → {OUTPUT_DIR / 'ICUSTAYS_TCN_Anomaly_Plot.png'}")

print("\n✅ TCN Autoencoder anomaly detection complete!")
