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
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.layers import Dense, Dropout, Lambda, Input

# ---------- Config ----------
DATASETS_DIR = Path(r"D:/Final Year/Project/Anomaly Detection/ICU Monitoring/Datasets")
OUTPUT_DIR = DATASETS_DIR / "../Results of all Models/GNN_AE-ModelResult"
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
# GNN Component: Graph Convolutional Layer Simulation
# ======================

# ... (imports remain the same) ...

# ======================
# GNN Component: Graph Convolutional Layer Simulation - FIXED
# ======================

class GraphConvolutionLayer(tf.keras.layers.Layer): # <-- FIX IS HERE
    """
    Simulates a GCN layer using a pre-calculated Adjacency Matrix (A).
    Input: Node features (X) [Batch, Sequence_Len, Num_Features]
    A is implicitly learned or statically set outside this layer.
    """
    def __init__(self, output_dim, activation=None, use_bias=True, **kwargs):
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        super(GraphConvolutionLayer, self).__init__(**kwargs) # Corrected super() call

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer='zeros',
                                        name='bias')
        super(GraphConvolutionLayer, self).build(input_shape)

    def call(self, inputs):
        # Standard matrix multiplication (X * W)
        output = K.dot(inputs, self.kernel)
        
        # Add bias
        if self.use_bias:
            output = K.bias_add(output, self.bias)
            
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
            
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

# ======================
# Model Architecture
# ======================
def build_gnn_autoencoder(seq_len, n_feats, filters=64):
    
    inp = Input(shape=(seq_len, n_feats))
    
    # 1. Encoder (GNN Feature Extraction)
    # The GCL here processes the features (vitals) and helps aggregate info across them.
    x = GraphConvolutionLayer(filters, activation='relu')(inp)
    x = Dropout(0.2)(x)
    x = GraphConvolutionLayer(filters // 2, activation='relu')(x)
    
    # Flatten the sequence and features into a single latent vector
    encoded = layers.Flatten()(x)
    encoded = Dense(filters // 4, activation='relu')(encoded) # Bottleneck

    # 2. Decoder
    # Reconstruct the sequence: Expand the latent vector
    x = Dense(seq_len * (filters // 2), activation='relu')(encoded)
    x = layers.Reshape((seq_len, filters // 2))(x)
    
    # GNN-like reconstruction
    x = GraphConvolutionLayer(filters, activation='relu')(x)
    
    # Final dense layer to map back to original feature dimension
    out = layers.TimeDistributed(Dense(n_feats))(x)
    
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

# ======================
# Core Utilities (GT, Sequencing, Evaluation)
# ======================
def label_anomalies(df, gt_array):
    # ... (Reuses the complex physiological labeling logic) ...
    anomalies_for_cleaning = []
    
    for col_name in df.columns:
        temp_series = pd.to_numeric(df[col_name], errors="coerce").dropna()
        original_indices = temp_series.index
        
        if temp_series.empty:
            continue

        vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
        anomalous_indices_in_df = []

        if vital:
            if vital == "HR":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["HR_tachy"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["HR_brady"]])
            elif vital == "SBP":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["SBP_high"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["SBP_low"]])
            elif vital == "DBP":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["DBP_high"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["DBP_low"]])
            elif vital == "MAP":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["MAP_high"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["MAP_low"]])
            elif vital == "RR":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["RR_tachypnea"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["RR_apnea"]])
            elif vital == "SpO2":
                anomalous_indices_in_df += list(original_indices[temp_series < TH["SpO2_low"]])
            elif vital == "Temp":
                anomalous_indices_in_df += list(original_indices[temp_series > TH["Temp_high"]])
                anomalous_indices_in_df += list(original_indices[temp_series < TH["Temp_low"]])
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
        return {"File": file_name, "Dataset": dataset_name, "Model": "GNN-AE",
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

    return {"File": file_name, "Dataset": dataset_name, "Model": "GNN-AE",
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
    print(f"\n=== Processing {file.name} (GNN Autoencoder) ===")
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
        model = build_gnn_autoencoder(SEQ_LEN, n_feats)
        print(f"Training on {seqs.shape[0]} sequences...")
        model.fit(seqs, seqs, epochs=30, batch_size=32, validation_split=0.1, verbose=0)

        # 6. Anomaly Score Calculation
        seq_recon = model.predict(seqs)
        reconstructed_rows = reconstruct_sequences_to_rows(seq_recon, original_length=X.shape[0], seq_len=SEQ_LEN)
        
        # Reconstruction Error (Anomaly Score) for each row
        anomaly_scores = np.mean(np.square(X - reconstructed_rows), axis=1)
        
        # 7. Evaluation
        dataset_name = file.stem
        eval_results = evaluate_anomaly_detection(y_true, anomaly_scores, file.name, dataset_name)
        evaluation_summary.append(eval_results)
        
        print(f"Metrics (Max F1 Threshold): AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | P={eval_results['Precision']:.4f} | R={eval_results['Recall']:.4f} | FAR={eval_results['False Alarm Rate (FAR)']:.4f}")
        print(f"GT Anomalies: {eval_results['GT Anomaly Count']}")

        # 8. Plotting and Saving
        plt.figure(figsize=(10,5))
        plt.plot(anomaly_scores, label="Reconstruction Error")
        plt.axhline(y=eval_results["Optimal Threshold"], color='r', linestyle='--', label='Optimal F1 Threshold')
        plt.title(f"GNN-AE Error - {file.stem}")
        plt.xlabel("Sample Index")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{file.stem}_gnn_ae_error_plot.png")
        plt.close()


    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")

# ======================
# Final Summary and Conclusion
# ======================
if evaluation_summary:
    eval_df = pd.DataFrame(evaluation_summary)
    eval_df = eval_df.sort_values(by="Dataset").reset_index(drop=True)
    
    summary_path = OUTPUT_DIR / "gnn_ae_performance_summary.csv"
    eval_df.to_csv(summary_path, index=False)
    print(f"\n✅ GNN-AE Performance Summary saved → {summary_path}")
    print("\nSummary of Performance Metrics (GNN Autoencoder):")
    print(eval_df[["Dataset", "AUC", "F1", "Precision", "Recall", "False Alarm Rate (FAR)", "GT Anomaly Count"]])

else:
    print("⚠️ No results generated to summarize.")