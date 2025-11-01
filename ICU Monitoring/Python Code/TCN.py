import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("\n=== Training TCN Autoencoder anomaly detection model on ICU.csv ===")

# ===================================
# 1️⃣ Setup paths
# ===================================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\TCN-ModelResult"
os.makedirs(results_dir, exist_ok=True)

# ===================================
# 2️⃣ Load and preprocess data
# ===================================
df = pd.read_csv(input_path)
features = ['Age', 'SysBP', 'Pulse']  # Same as Isolation Forest / LOF / SVM
df = df[features].dropna().reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Reshape for Conv1D input: (samples, timesteps, features)
# Here each record is treated as a short time sequence of length=1
X_scaled = np.expand_dims(X_scaled, axis=1)

# ===================================
# 3️⃣ Build TCN Autoencoder
# ===================================
def build_tcn_autoencoder(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv1D(32, kernel_size=3, padding="same", activation="relu", dilation_rate=1)(inputs)
    x = Conv1D(16, kernel_size=3, padding="same", activation="relu", dilation_rate=2)(x)
    encoded = Conv1D(8, kernel_size=3, padding="same", activation="relu", dilation_rate=4)(x)

    # Decoder
    x = Conv1D(16, kernel_size=3, padding="same", activation="relu", dilation_rate=2)(encoded)
    x = Conv1D(32, kernel_size=3, padding="same", activation="relu", dilation_rate=1)(x)
    decoded = Conv1D(input_shape[-1], kernel_size=3, padding="same")(x)

    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

model = build_tcn_autoencoder(X_scaled.shape[1:])
model.summary()

# ===================================
# 4️⃣ Train Model
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

# Threshold: Mean + 2×Std
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold

df['Reconstruction_Error'] = mse
df['Anomaly'] = anomalies.astype(int)

# ===================================
# 6️⃣ Summary
# ===================================
num_anomalies = df['Anomaly'].sum()
summary_text = (
    f"=== TCN Autoencoder Anomaly Detection Summary ===\n"
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
results_csv = os.path.join(results_dir, "TCN_Autoencoder_Results.csv")
summary_txt = os.path.join(results_dir, "TCN_Autoencoder_Summary.csv")
plot_path = os.path.join(results_dir, "TCN_Autoencoder_Plot.png")

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
plt.title("TCN Autoencoder Anomaly Detection on ICU Data")
plt.grid(True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# ===================================
# 9️⃣ Done
# ===================================
print(f"✅ TCN Autoencoder training completed successfully.")
print(f"   - Results: {results_csv}")
print(f"   - Summary: {summary_txt}")
print(f"   - Plot: {plot_path}")
