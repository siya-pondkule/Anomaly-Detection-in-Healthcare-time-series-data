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
from tensorflow.keras.layers import Lambda
from tcn import TCN 

# ---------- Config ----------
DATASETS_DIR = Path(r"D:/Final Year/Project/Anomaly Detection/ICU Monitoring/Datasets")
OUTPUT_DIR = DATASETS_DIR / "../Results of all Models/TCN_AE-ModelResult"
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

# ======================
# Detect and LABEL anomalies (Ground Truth)
# ======================
def label_anomalies(df, gt_array):
    """Detects physiological anomalies and updates the Ground Truth array."""
    anomalies_for_cleaning = []
    
    for col_name in df.columns:
        temp_series = pd.to_numeric(df[col_name], errors="coerce").dropna()
        original_indices = temp_series.index
        
        if temp_series.empty:
            continue

        vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
        anomalous_indices_in_df = []

        # --- ANOMALY DETECTION LOGIC (based on TH) ---
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
            # generic z-score detection
            z = (temp_series - temp_series.mean()) / temp_series.std(ddof=0)
            anomalous_indices_in_df += list(original_indices[np.abs(z) > 3])

        # Update the Ground Truth array (1 for anomaly)
        for idx in anomalous_indices_in_df:
            if idx < len(gt_array):
                gt_array[idx] = 1
                anomalies_for_cleaning.append({"vital": col_name, "index": idx})
                
    return anomalies_for_cleaning

# --- Sequence and Reconstruction Helpers (Reused from previous steps) ---
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

# ======================
# Performance Metrics Calculation
# ======================
def evaluate_anomaly_detection(y_true, anomaly_scores, file_name, dataset_name):
    """
    Calculates all required performance metrics (AUC, F1, P, R, FAR) 
    using the reconstruction MSE scores (higher = more anomalous).
    """
    
    scores = anomaly_scores
    
    # 1. Check for valid ground truth
    if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return {
            "File": file_name, "Dataset": dataset_name, "Model": "TCN-AE",
            "AUC": np.nan, "F1": 0.0, "Precision": 0.0, "Recall": 0.0, 
            "False Alarm Rate (FAR)": 0.1, "Optimal Threshold": np.nan, "GT Anomaly Count": np.sum(y_true)
        }

    # 2. AUC 
    try:
        if np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
            auc = roc_auc_score(y_true, scores)
        else:
            auc = np.nan
    except ValueError:
        auc = np.nan

    # 3. Optimal Threshold Search (Maximize F1 Score)
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
    
    # 4. Precision, Recall, F1, False Alarm Rate (FAR)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    precision = precision_score(y_true, y_pred_best, zero_division=0)
    recall = recall_score(y_true, y_pred_best, zero_division=0)
    f1 = best_f1
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "File": file_name,
        "Dataset": dataset_name,
        "Model": "TCN-AE",
        "AUC": auc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "False Alarm Rate (FAR)": false_alarm_rate,
        "Optimal Threshold": best_threshold,
        "GT Anomaly Count": np.sum(y_true)
    }

# ---------- TCN Autoencoder Model ----------
def build_tcn_autoencoder(seq_len, n_feats):
    inp = layers.Input(shape=(seq_len, n_feats))
    
    # Encoder - TCN Stack
    # TCN returns sequences by default
    x = TCN(
        nb_filters=64, kernel_size=3, nb_stacks=1, dilations=[1, 2, 4, 8], 
        padding='causal', use_skip_connections=False, return_sequences=True, activation='relu'
    )(inp)
    
    # Compress the sequence to the final hidden state (Lambda layer used for sequence reduction)
    # The latent vector is the output of the last time step
    encoded = Lambda(lambda x: x[:, -1, :])(x) 

    # Latent Space and Sequence Repetition
    repeated = layers.RepeatVector(seq_len)(encoded)
    
    # Decoder - TCN Stack (Must use 'same' padding for output to match input shape)
    decoded = TCN(
        nb_filters=64, kernel_size=3, nb_stacks=1, dilations=[1, 2, 4, 8], 
        padding='same', use_skip_connections=False, return_sequences=True, activation='relu'
    )(repeated)

    # Output (Time Distributed Dense layer)
    out = layers.TimeDistributed(layers.Dense(n_feats))(decoded)
    
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model


# ---------- Main Processing Loop ----------
SEQ_LEN = 10 
evaluation_summary = []

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} (TCN Autoencoder) ===")
    try:
        df_original = pd.read_csv(file, low_memory=False)
        df = df_original.copy()
        
        # 1. Initial Cleaning and Index Reset
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(axis=1, how='all').reset_index(drop=True)
        if df.empty:
            print("No numeric columns.")
            continue

        # 2. --- GROUND TRUTH GENERATION ---
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
        if len(X) < SEQ_LEN:
            print(f"Not enough rows ({len(X)}) for sequence modelling — skipping.")
            continue

        seqs = create_sequences(X, seq_len=SEQ_LEN)
        n_feats = seqs.shape[2]
        y_true = gt_array_full[:len(X)]

        # 5. Model Training
        model = build_tcn_autoencoder(SEQ_LEN, n_feats)
        print(f"Training on {seqs.shape[0]} sequences...")
        model.fit(seqs, seqs, epochs=30, batch_size=32, validation_split=0.1, verbose=0)

        # 6. Anomaly Score Calculation
        seq_recon = model.predict(seqs)
        reconstructed_rows = reconstruct_sequences_to_rows(seq_recon, original_length=X.shape[0], seq_len=SEQ_LEN)
        
        # Reconstruction Error (Anomaly Score) for each row
        anomaly_scores = np.mean(np.square(X - reconstructed_rows), axis=1)
        
        # 7. Evaluation against Ground Truth
        dataset_name = file.stem
        eval_results = evaluate_anomaly_detection(y_true, anomaly_scores, file.name, dataset_name)
        evaluation_summary.append(eval_results)
        
        print(f"Metrics (Max F1 Threshold): AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | P={eval_results['Precision']:.4f} | R={eval_results['Recall']:.4f} | FAR={eval_results['False Alarm Rate (FAR)']:.4f}")
        print(f"GT Anomalies: {eval_results['GT Anomaly Count']}")

        # 8. Reconstruction Metrics (for sanity check)
        overall_mse = anomaly_scores.mean()
        accuracy = 100 * (1 - overall_mse / np.var(X))
        print(f"Reconstruction Metrics → Approx. Accuracy: {accuracy:.2f}% | Avg MSE: {overall_mse:.6f}")

        # 9. Plotting and Saving
        recon_inv = scaler.inverse_transform(reconstructed_rows)
        df_corrected = pd.DataFrame(recon_inv, columns=df_clean.columns)
        corrected_path = OUTPUT_DIR / f"{file.stem}_corrected_tcn_ae.csv"
        df_corrected.to_csv(corrected_path, index=False)
        
        # Simple error plot
        plt.figure(figsize=(10,5))
        plt.plot(anomaly_scores, label="Reconstruction Error")
        plt.axhline(y=eval_results["Optimal Threshold"], color='r', linestyle='--', label='Optimal F1 Threshold')
        plt.title(f"TCN-AE Error - {file.stem}")
        plt.xlabel("Sample Index")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{file.stem}_tcn_ae_error_plot.png")
        plt.close()


    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")

# ======================
# Final Summary and Conclusion
# ======================
if evaluation_summary:
    eval_df = pd.DataFrame(evaluation_summary)
    eval_df = eval_df.sort_values(by="Dataset").reset_index(drop=True)
    
    summary_path = OUTPUT_DIR / "tcn_ae_performance_summary.csv"
    eval_df.to_csv(summary_path, index=False)
    print(f"\nTCN-AE Performance Summary saved → {summary_path}")
    print("\nSummary of Performance Metrics (TCN Autoencoder):")
    print(eval_df[["Dataset", "AUC", "F1", "Precision", "Recall", "False Alarm Rate (FAR)", "GT Anomaly Count"]])

else:
    print("No results generated to summarize.")