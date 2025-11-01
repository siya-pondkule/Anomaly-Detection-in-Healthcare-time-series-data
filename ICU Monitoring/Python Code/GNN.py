import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras import layers, models
from spektral.layers import GCNConv
from spektral.utils import normalized_adjacency, sp_matrix_to_sp_tensor

# ======================
# Paths
# ======================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\GNN-ModelResults")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Parameters
# ======================
SIMILARITY_THRESHOLD = 0.9
EPOCHS = 100

print(f"\n=== Training GNN model on {DATA_PATH.name} ===")

# ======================
# Load and preprocess
# ======================
df = pd.read_csv(DATA_PATH)
df = df[["SysBP", "Pulse"]].dropna().reset_index(drop=True)

scaler = StandardScaler()
X = scaler.fit_transform(df.values)

# ======================
# Build normalized adjacency (cosine similarity)
# ======================
cos_sim = cosine_similarity(X)
A = (cos_sim > SIMILARITY_THRESHOLD).astype(np.float32)
np.fill_diagonal(A, 1.0)

# Normalize adjacency (important for stability)
A_norm = normalized_adjacency(A)
# Convert to SparseTensor
A_sparse = sp_matrix_to_sp_tensor(A_norm)

# Convert X (node features) to tensor
X_tf = tf.convert_to_tensor(X, dtype=tf.float32)

# ======================
# Build GNN Autoencoder using Keras Functional API
# ======================
n_features = X.shape[1]
n_nodes = X_tf.shape[0] # The number of nodes
n_hidden = 32

# 1. Define Inputs
# x_in for node features (2 features: SysBP, Pulse)
x_in = tf.keras.Input(shape=(n_features,), dtype=tf.float32, name="features")
# a_in for adjacency matrix (N x N, sparse=True for efficiency)
# FIX: Explicitly set the shape to (N, N) for the sparse adjacency matrix
a_in = tf.keras.Input(shape=(n_nodes, n_nodes), sparse=True, dtype=tf.float32, name="adjacency") 

# 2. Encoder
h = GCNConv(n_hidden, activation='relu')([x_in, a_in])
h = GCNConv(n_hidden // 2, activation='relu')([h, a_in])

# 3. Decoder
output = layers.Dense(n_features, activation='linear')(h)

# 4. Create Model
model = tf.keras.Model(inputs=[x_in, a_in], outputs=output)
model.compile(optimizer='adam', loss='mse')

# ======================
# Train
# ======================
print("Training started...")
history = model.fit(x=[X_tf, A_sparse],y=X_tf,epochs=EPOCHS,batch_size=X_tf.shape[0],verbose=1)
print(f"Training complete | Final Loss: {history.history['loss'][-1]:.6f}")

# ======================
# Reconstruction & anomaly detection
# ======================
X_pred = model([X_tf, A_sparse]).numpy()
mse_scores = np.mean(np.square(X - X_pred), axis=1)
threshold = np.percentile(mse_scores, 95)
anomaly_indices = np.where(mse_scores >= threshold)[0]
print(f"Total anomalies detected: {len(anomaly_indices)}")

# ======================
# Save results
# ======================
summary_df = pd.DataFrame({"Record Index": anomaly_indices,"Reconstruction Error": mse_scores[anomaly_indices]
})
summary_df.to_csv(OUTPUT_DIR / "GNN_Anomaly_Summary.csv", index=False)
print(f"🧾 Anomaly summary saved → {OUTPUT_DIR / 'GNN_Anomaly_Summary.csv'}")

# ======================
# Visualization
# ======================
plt.figure(figsize=(12, 6))
plt.plot(df["SysBP"], label="SysBP", alpha=0.7)
plt.plot(df["Pulse"], label="Pulse", alpha=0.7)
for idx in anomaly_indices:plt.scatter(idx, df.loc[idx, "SysBP"], color="red", marker="x", s=50)
plt.scatter(idx, df.loc[idx, "Pulse"], color="red", marker="x", s=50)

plt.title("ICU Anomaly Detection using Graph Neural Network (GNN Autoencoder)")
plt.xlabel("Record Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ICU_GNN_AnomalyPlot.png")
plt.close()

print("\n✅ GNN Model training, evaluation, and visualization completed successfully.")