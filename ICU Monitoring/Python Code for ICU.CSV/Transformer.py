import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("\n=== Training Transformer Autoencoder anomaly detection model on ICU.csv ===")

# ===================================
# 1️⃣ Setup paths
# ===================================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\Transformer-ModelResult"
os.makedirs(results_dir, exist_ok=True)

# ===================================
# 2️⃣ Load and preprocess data
# ===================================
df = pd.read_csv(input_path)
features = ['Age', 'SysBP', 'Pulse']  # Same as previous models
df = df[features].dropna().reset_index(drop=True)

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Transformer expects sequence input — reshape accordingly
# Treat each sample as a "sequence" of 1 timestep for now
X_scaled = np.expand_dims(X_scaled, axis=1)

# ===================================
# 3️⃣ Define Transformer Autoencoder
# ===================================
def transformer_block(inputs, num_heads, ff_dim, dropout=0.1):
    # Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    # Feed-forward network
    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn = Dropout(dropout)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    return out2

def build_transformer_autoencoder(input_shape, num_heads=2, ff_dim=64):
    inputs = Input(shape=input_shape)

    # Encoder
    x = transformer_block(inputs, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)
    encoded = Dense(16, activation='relu')(x)

    # Decoder
    x = transformer_block(encoded, num_heads=num_heads, ff_dim=ff_dim)
    decoded = Dense(input_shape[-1], activation=None)(x)

    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

model = build_transformer_autoencoder(X_scaled.shape[1:])
model.summary()

# ===================================
# 4️⃣ Train the model
# ===================================
EPOCHS = 50
BATCH_SIZE = 32

history = model.fit(
    X_scaled, X_scaled,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# ===================================
# 5️⃣ Compute reconstruction errors
# ===================================
reconstructions = model.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=(1, 2))

threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

df['Reconstruction_Error'] = mse
df['Anomaly'] = anomalies.astype(int)

# ===================================
# 6️⃣ Summary
# ===================================
num_anomalies = df['Anomaly'].sum()
summary_text = (
    f"=== Transformer Autoencoder Anomaly Detection Summary ===\n"
    f"Total samples: {len(df)}\n"
    f"Detected anomalies: {num_anomalies}\n"
    f"Normal points: {len(df) - num_anomalies}\n"
    f"Features used: {features}\n"
    f"Reconstruction error threshold: {threshold:.6f}\n"
)

print("\n" + summary_text)

# ===================================
# 7️⃣ Save results
# ===================================
results_csv = os.path.join(results_dir, "Transformer_Autoencoder_Results.csv")
summary_txt = os.path.join(results_dir, "Transformer_Autoencoder_Summary.csv")
plot_path = os.path.join(results_dir, "Transformer_Autoencoder_Plot.png")

df.to_csv(results_csv, index=False)
with open(summary_txt, "w") as f:
    f.write(summary_text)

# ===================================
# 8️⃣ Visualization
# ===================================
plt.figure(figsize=(8, 6))
plt.scatter(df['SysBP'], df['Pulse'], c=df['Anomaly'], cmap='coolwarm', edgecolors='k')
plt.xlabel("Systolic Blood Pressure (SysBP)")
plt.ylabel("Pulse")
plt.title("Transformer Autoencoder Anomaly Detection on ICU Data")
plt.grid(True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# ===================================
# 9️⃣ Done
# ===================================
print(f"✅ Transformer Autoencoder training completed successfully.")
print(f"   - Results: {results_csv}")
print(f"   - Summary: {summary_txt}")
print(f"   - Plot: {plot_path}")
