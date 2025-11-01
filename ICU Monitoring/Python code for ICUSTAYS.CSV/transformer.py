import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, MultiHeadAttention, Flatten, Reshape, Add
from tensorflow.keras.callbacks import EarlyStopping

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\ICUSTAYS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for ICUSTATYS\Transformer-ModelResults")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of ICUSTAYS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Load and preprocess data
# ======================
print(f"\n=== Processing {DATA_PATH.name} for Transformer Autoencoder ===")
df = pd.read_csv(DATA_PATH)

# Keep only numeric columns and drop identifiers
drop_cols = [c for c in df.columns if any(x in c.lower() for x in ["id", "time", "date", "unit"])]
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
df_numeric = df_numeric.dropna().reset_index(drop=True)

if df_numeric.empty:
    raise ValueError("No usable numeric columns found in ICUSTAYS.csv for Transformer model.")

print(f"Using columns for anomaly detection: {list(df_numeric.columns)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# ======================
# Create sequences for transformer
# ======================
time_steps = 10
def create_sequences(data, time_steps=10):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

X_seq = create_sequences(X_scaled, time_steps)
print(f"Shape of input data for transformer: {X_seq.shape}")

# ======================
# Transformer Encoder Block
# ======================
def transformer_block(x, num_heads, ff_dim, dropout_rate):
    # Multi-head attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dense(x.shape[-1])(ff_output)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# ======================
# Build Transformer Autoencoder
# ======================
def build_transformer_autoencoder(timesteps, n_features, num_heads=4, ff_dim=64, dropout_rate=0.1):
    inputs = Input(shape=(timesteps, n_features))

    # Encoder
    x = transformer_block(inputs, num_heads, ff_dim, dropout_rate)
    x = transformer_block(x, num_heads, ff_dim, dropout_rate)
    encoded = Flatten()(x)
    bottleneck = Dense(64, activation='relu')(encoded)

    # Decoder
    x = Dense(timesteps * n_features, activation='relu')(bottleneck)
    x = Reshape((timesteps, n_features))(x)
    x = transformer_block(x, num_heads, ff_dim, dropout_rate)
    outputs = Dense(n_features, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_transformer_autoencoder(X_seq.shape[1], X_seq.shape[2])
model.summary()

# ======================
# Train Transformer Autoencoder
# ======================
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\nTraining Transformer Autoencoder...")
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
mse = np.mean(np.mean(np.square(X_seq - X_pred), axis=2), axis=1)

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
results_file = OUTPUT_DIR / "ICUSTAYS_Transformer_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

# ======================
# Save summary
# ======================
summary_data = {
    "Total Sequences": len(mse),
    "Detected Anomalies": len(anomalies),
    "Anomaly Percentage (%)": round(100 * len(anomalies) / len(mse), 2)
}
summary_df = pd.DataFrame([summary_data])
summary_file = SUMMARY_DIR / "Transformer_ICUSTAYS_Summary.csv"
summary_df.to_csv(summary_file, index=False)
print(f"Summary saved → {summary_file}")

# ======================
# Plot reconstruction error
# ======================
plt.figure(figsize=(12, 6))
plt.plot(mse, label="Reconstruction Error", color="blue", alpha=0.7)
plt.scatter(anomalies, mse[anomalies], color="red", label="Detected Anomalies", marker="x")
plt.axhline(threshold, color="orange", linestyle="--", label="Threshold")
plt.title("Transformer Autoencoder Anomaly Detection - ICUSTAYS.csv")
plt.xlabel("Sequence Index")
plt.ylabel("Reconstruction Error (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICUSTAYS_Transformer_Anomaly_Plot.png")
plt.close()

print(f"Plot saved → {OUTPUT_DIR / 'ICUSTAYS_Transformer_Anomaly_Plot.png'}")

print("\n✅ Transformer Autoencoder anomaly detection complete!")
