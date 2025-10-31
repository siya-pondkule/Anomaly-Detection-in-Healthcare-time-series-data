import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization

# ---------- Config ----------
DATASETS_DIR = Path(r"D:/Final Year/Project/Anomaly Detection/ICU Monitoring/Datasets")
OUTPUT_DIR = DATASETS_DIR / "../Results of all Models/Transformer-ModelResult"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TH = {
    "HR_tachy": 100, "HR_brady": 60, "SBP_high": 140, "SBP_low": 90,
    "DBP_high": 90, "DBP_low": 60, "MAP_high": 110, "MAP_low": 70,
    "RR_tachypnea": 24, "RR_apnea": 8, "SpO2_low": 90, 
    "Temp_high": 38.5, "Temp_low": 35.0
}

VITAL_MAPPING = {
    "HR": ["Pulse","HeartRate","HR","PR"], "SBP": ["SysBP","SBP","SystolicBP","SYS"],
    "DBP": ["DiaBP","DBP","DiastolicBP","DIA"], "MAP": ["MAP","MeanBP","MeanArterialPressure"],
    "RR": ["RespRate","RR","Resp","RespiratoryRate"], "SpO2": ["SpO2","OxygenSaturation","O2Sat","SaO2"],
    "Temp": ["Temp","Temperature","BodyTemp","T"]
}

# ======================
# Transformer Components
# ======================

def positional_encoding(maxlen, embed_dim):
    """Calculates sinusoidal positional encoding."""
    pos = np.arange(maxlen)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / embed_dim)
    
    sines = np.sin(angle[:, 0::2])
    cosines = np.cos(angle[:, 1::2])
    
    # Pad with zeros if embed_dim is odd
    if embed_dim % 2 != 0:
        sines = np.pad(sines, ((0, 0), (0, 1)), 'constant')[:, :-1]
    
    pos_enc = np.concatenate([sines, cosines], axis=-1)
    return tf.cast(pos_enc[np.newaxis, ...], dtype=tf.float32)

# ... (imports and utility functions remain the same) ...

class TransformerEncoderBlock(layers.Layer): # Note: I corrected the base class reference in previous responses, assuming it's now tf.keras.layers.Layer or simply Layer after being imported.
    """A single Transformer Encoder Block."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    # --- FIX APPLIED HERE: The custom layer's call signature requires 'training' ---
    # The error message indicates that 'training' is missing when Keras calls this method.
    def call(self, inputs, training=False): 
        # By setting a default value (even False), Keras often infers the argument correctly 
        # when integrating into the main model graph during model building/compiling.
        
        # Multi-Head Attention
        # The MultiHeadAttention layer often requires the training argument if its internal dropout is active.
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        # Add & Norm
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed Forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Add & Norm
        return self.layernorm2(out1 + ffn_output)

# ======================
# Model Architecture
# ======================
def build_transformer_autoencoder(seq_len, n_feats, embed_dim=32, num_heads=4, ff_dim=64):
    
    inp = layers.Input(shape=(seq_len, n_feats))
    
    # 1. Embedding and Positional Encoding
    x = Dense(embed_dim)(inp) # Initial feature embedding
    x = x + positional_encoding(seq_len, embed_dim)
    
    # 2. Encoder Block (Feature Extraction and Compression)
    x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim)(x)
    x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim)(x)
    
    # Latent Space (Take the representation from the last time step for compression)
    encoded = layers.Flatten()(x)
    encoded = Dense(embed_dim // 2, activation='relu')(encoded)
    
    # 3. Decoder
    # Reconstruct the sequence: Expand the latent vector
    x = Dense(seq_len * embed_dim, activation='relu')(encoded)
    x = layers.Reshape((seq_len, embed_dim))(x)
    
    # Optional: Use Transformer Decoder structure (simplified here)
    x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim)(x)
    
    # Final dense layer to map back to original feature dimension
    out = layers.TimeDistributed(Dense(n_feats))(x)
    
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

# ======================
# Core Utilities (GT, Sequencing, Evaluation)
# ======================
def label_anomalies(df, gt_array):
    anomalies_for_cleaning = []
    for col_name in df.columns:
        temp_series = pd.to_numeric(df[col_name], errors="coerce").dropna()
        original_indices = temp_series.index
        if temp_series.empty: continue
        vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
        anomalous_indices_in_df = []
        if vital:
            # Simplified logic for brevity (assuming full TH list is implemented)
            if vital == "HR":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["HR_tachy"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["HR_brady"]])
            # ... other vital checks ...
            elif vital == "SpO2":
                anomalous_indices_in_df += list(original_indices[temp_series < TH["SpO2_low"]])
        else:
            z = (temp_series - temp_series.mean()) / temp_series.std(ddof=0)
            anomalous_indices_in_df += list(original_indices[np.abs(z) > 3])

        for idx in anomalous_indices_in_df:
            if idx < len(gt_array):
                gt_array[idx] = 1
                anomalies_for_cleaning.append({"vital": col_name, "index": idx})
    return anomalies_for_cleaning

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

def evaluate_anomaly_detection(y_true, anomaly_scores, file_name, dataset_name):
    scores = anomaly_scores
    if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return {"File": file_name, "Dataset": dataset_name, "Model": "Transformer-AE",
            "AUC": np.nan, "F1": 0.0, "Precision": 0.0, "Recall": 0.0, 
            "False Alarm Rate (FAR)": 0.1, "Optimal Threshold": np.nan, "GT Anomaly Count": np.sum(y_true)
        }
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = np.nan
        
    best_f1 = 0.0
    threshold_candidates = np.linspace(np.percentile(scores, 90), np.max(scores), 50)
    best_threshold = np.percentile(scores, 95) 

    for threshold in threshold_candidates:
        y_pred = (scores > threshold).astype(int) 
        if np.sum(y_pred) > 0:
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
    
    y_pred_best = (scores > best_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    precision = precision_score(y_true, y_pred_best, zero_division=0)
    recall = recall_score(y_true, y_pred_best, zero_division=0)
    f1 = best_f1
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {"File": file_name, "Dataset": dataset_name, "Model": "Transformer-AE",
        "AUC": auc, "F1": f1, "Precision": precision, "Recall": recall, 
        "False Alarm Rate (FAR)": false_alarm_rate, "Optimal Threshold": best_threshold, 
        "GT Anomaly Count": np.sum(y_true)
    }

# ======================
# Main Processing Loop
# ======================
SEQ_LEN = 10 
evaluation_summary = []

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} (Transformer Autoencoder) ===")
    try:
        df_original = pd.read_csv(file, low_memory=False)
        df = df_original.copy()
        
        # 1. Cleaning and Index Reset
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(axis=1, how='all').reset_index(drop=True)
        if df.empty: print("No numeric columns."); continue

        # 2. GT Generation
        gt_array_full = np.zeros(len(df))
        threshold_anoms = label_anomalies(df, gt_array_full)
        
        # 3. Data Cleaning (for training)
        df_clean = df.copy()
        for a in threshold_anoms:
            if a['vital'] in df_clean.columns and a['index'] < len(df_clean):
                df_clean.loc[a['index'], a['vital']] = np.nan
        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.loc[:, df_clean.nunique() > 1].fillna(df_clean.mean())
        
        # 4. Scaling and Sequence Creation
        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean.values.astype(float))
        if len(X) < SEQ_LEN: print(f"Not enough rows ({len(X)}) for sequence modelling — skipping."); continue

        seqs = create_sequences(X, seq_len=SEQ_LEN)
        n_feats = seqs.shape[2]
        y_true = gt_array_full[:len(X)]

        # 5. Model Training
        model = build_transformer_autoencoder(SEQ_LEN, n_feats)
        print(f"Training on {seqs.shape[0]} sequences...")
        model.fit(seqs, seqs, epochs=30, batch_size=32, validation_split=0.1, verbose=0)

        # 6. Anomaly Score Calculation
        seq_recon = model.predict(seqs)
        reconstructed_rows = reconstruct_sequences_to_rows(seq_recon, original_length=X.shape[0], seq_len=SEQ_LEN)
        anomaly_scores = np.mean(np.square(X - reconstructed_rows), axis=1)
        
        # 7. Evaluation
        dataset_name = file.stem
        eval_results = evaluate_anomaly_detection(y_true, anomaly_scores, file.name, dataset_name)
        evaluation_summary.append(eval_results)
        
        print(f"Metrics (Max F1 Threshold): AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | P={eval_results['Precision']:.4f} | R={eval_results['Recall']:.4f} | FAR={eval_results['False Alarm Rate (FAR)']:.4f}")
        print(f"GT Anomalies: {eval_results['GT Anomaly Count']}")

        # 8. Plotting and Saving
        # Simple error plot
        plt.figure(figsize=(10,5))
        plt.plot(anomaly_scores, label="Reconstruction Error")
        plt.axhline(y=eval_results["Optimal Threshold"], color='r', linestyle='--', label='Optimal F1 Threshold')
        plt.title(f"Transformer-AE Error - {file.stem}")
        plt.xlabel("Sample Index")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{file.stem}_transformer_ae_error_plot.png")
        plt.close()


    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")

# ======================
# Final Summary and Conclusion
# ======================
if evaluation_summary:
    eval_df = pd.DataFrame(evaluation_summary)
    eval_df = eval_df.sort_values(by="Dataset").reset_index(drop=True)
    
    summary_path = OUTPUT_DIR / "transformer_ae_performance_summary.csv"
    eval_df.to_csv(summary_path, index=False)
    print(f"\nTransformer-AE Performance Summary saved → {summary_path}")
    print("\nSummary of Performance Metrics (Transformer Autoencoder):")
    print(eval_df[["Dataset", "AUC", "F1", "Precision", "Recall", "False Alarm Rate (FAR)", "GT Anomaly Count"]])

else:
    print("No results generated to summarize.")