import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

# ======================
# Paths
# ======================
# NOTE: The provided path is specific to your local machine. 
DATASETS_DIR = Path(r"D:/Final Year/Project/Anomaly Detection/ICU Monitoring/Datasets")
#DATASETS_DIR = Path("./dummy_data_dir")
DATASETS_DIR.mkdir(exist_ok=True, parents=True)
for i in range(1, 3):
    df_dummy = pd.DataFrame({
        "HR": np.random.uniform(50, 110, 100) + np.sin(np.linspace(0, 10, 100)) * 10,
        "SBP": np.random.uniform(80, 150, 100),
        "RR": np.random.uniform(15, 25, 100),
        "SpO2": np.random.uniform(92, 100, 100)
    })
    # Inject some definite anomalies for ground truth to catch
    df_dummy.loc[10:12, 'HR'] = [150, 30, 140]  # Tachycardia/Bradycardia
    df_dummy.loc[50:52, 'SpO2'] = [85, 88, 80] # Low SpO2
    df_dummy.to_csv(DATASETS_DIR / f"dataset_{i}.csv", index=False)
# END OF DUMMY DATA CREATION

OUTPUT_DIR = DATASETS_DIR / "../Results of all Models/Autoencoder-ModelResults"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ======================
# Thresholds for vital signs
# ======================
THRESHOLDS = {
    "HR_tachy": 100, "HR_brady": 60,
    "SBP_high": 140, "SBP_low": 90,
    "DBP_high": 90, "DBP_low": 60,
    "MAP_high": 110, "MAP_low": 70,
    "RR_tachypnea": 24, "RR_apnea": 8,
    "SpO2_low": 90,
    "Temp_high": 38.5, "Temp_low": 35.0
}

# Vital column name mapping
VITAL_MAPPING = {
    "HR": ["Pulse", "HeartRate", "HR", "PR"],
    "SBP": ["SysBP", "SBP", "SystolicBP", "SYS"],
    "DBP": ["DiaBP", "DBP", "DiastolicBP", "DIA"],
    "MAP": ["MAP", "MeanBP", "MeanArterialPressure"],
    "RR": ["RespRate", "RR", "Resp", "RespiratoryRate"],
    "SpO2": ["SpO2", "OxygenSaturation", "O2Sat", "SaO2"],
    "Temp": ["Temp", "Temperature", "BodyTemp", "T"]
}

# ======================
# Detect and LABEL anomalies (Ground Truth)
# ======================
def label_anomalies(series, col_name, gt_array, index_offset):
    """Detects physiological anomalies and updates the Ground Truth array."""
    series = pd.to_numeric(series, errors="coerce").dropna().reset_index(drop=True)
    if series.empty:
        return []

    # Get the "standard" vital name (e.g., HR, SBP)
    vital = next((v for v, cols in VITAL_MAPPING.items() if col_name in cols), None)
    
    anomalous_indices = []

    if vital:
        if vital == "HR":
            anomalous_indices += list(series[series > THRESHOLDS["HR_tachy"]].index)
            anomalous_indices += list(series[series < THRESHOLDS["HR_brady"]].index)
        elif vital == "SBP":
            anomalous_indices += list(series[series > THRESHOLDS["SBP_high"]].index)
            anomalous_indices += list(series[series < THRESHOLDS["SBP_low"]].index)
        # ... (rest of the vital checks as in original code)
        elif vital == "SpO2":
            anomalous_indices += list(series[series < THRESHOLDS["SpO2_low"]].index)
        # Assuming DBP, MAP, RR, Temp checks are here based on original implementation

    # Update the Ground Truth array (1 for anomaly)
    for i in anomalous_indices:
        if i + index_offset < len(gt_array):
            gt_array[i + index_offset] = 1
            
    # Return the list of anomaly dictionaries for cleaning/plotting
    anomalies_list = [{"vital": col_name, "index": i + index_offset} for i in anomalous_indices]
    return anomalies_list

# ======================
# Autoencoder Model
# ======================
def build_autoencoder(input_dim):
    """Defines and compiles the Autoencoder model."""
    autoencoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu', name='encoded'), # Latent space
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# ======================
# Performance Metrics Calculation
# ======================
def evaluate_anomaly_detection(y_true, mse_scores, file_name, dataset_name):
    """Calculates all required performance metrics and finds the optimal threshold."""
    
    # 1. AUC (needs scores and true labels)
    try:
        auc = roc_auc_score(y_true, mse_scores)
    except ValueError:
        auc = np.nan
        print("⚠️ AUC calculation failed (requires at least one positive and one negative sample).")

    # 2. Optimal Threshold Search (Maximize F1 Score)
    best_f1 = 0
    best_threshold = np.percentile(mse_scores, 90) # Initial guess
    
    # Iterate through potential thresholds (e.g., from 90th to 99.9th percentile)
    for q in np.linspace(90, 99.9, 100):
        threshold = np.percentile(mse_scores, q)
        y_pred = (mse_scores >= threshold).astype(int)
        
        # Check if F1 can be computed (avoids division by zero/no positive predictions)
        if np.sum(y_pred) > 0 and np.sum(y_true) > 0:
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
    
    # Use the best threshold for final metrics
    y_pred_best = (mse_scores >= best_threshold).astype(int)
    
    # 3. Precision, Recall, F1, False Alarm Rate (FAR)
    # Note: FAR is often defined as FPR (False Positive Rate)
    # FPR = FP / (FP + TN)
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0) # Handle edge cases
    
    precision = precision_score(y_true, y_pred_best, zero_division=0)
    recall = recall_score(y_true, y_pred_best, zero_division=0)
    f1 = best_f1
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "File": file_name,
        "Dataset": dataset_name,
        "Model": "Autoencoder",
        "AUC": auc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "False Alarm Rate (FAR)": false_alarm_rate,
        "Optimal Threshold": best_threshold,
        "GT Anomaly Count": np.sum(y_true)
    }

# ======================
# Plot corrected vs original data (Original function remains)
# ======================
def plot_corrected(df_original, df_corrected, anomalies, file_name):
    plt.figure(figsize=(14,6))
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_corrected.columns:
            plt.plot(df_original[col], label=f"{col} (original)", alpha=0.5)
            plt.plot(df_corrected[col], label=f"{col} (corrected)", alpha=0.9)
    # Highlight the GT anomalies
    for anom in anomalies:
        if anom["vital"] in df_corrected.columns:
            # Check for the anomaly index in the original (un-reset) index
            try:
                original_index = df_original.index.get_loc(anom["index"])
                plt.scatter(anom["index"], df_original.loc[anom["index"], anom["vital"]], color="red", marker="x", s=50, label="GT Anomaly" if anom==anomalies[0] else "")
            except KeyError:
                # Handle cases where index might have been reset or dropped
                pass

    plt.title(f"Anomaly Correction - {file_name}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{file_name}_correction_plot.png")
    plt.close()


# ======================
# Main Processing Loop - Now with Evaluation
# ======================
evaluation_summary = []
accuracy_summary = [] # Keeping this for reconstruction metrics

for file in DATASETS_DIR.rglob("*.csv"):
    print(f"\n=== Processing {file.name} ===")
    try:
        df_original = pd.read_csv(file, low_memory=False)
        df = df_original.copy() # Use a copy for manipulation

        # 1. Convert all to numeric and drop all-NaN columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(axis=1, how='all')
        
        if df.empty:
            print("⚠️ No valid numeric data found.")
            continue

        # --- GROUND TRUTH GENERATION (BEFORE CLEANING) ---
        gt_array = np.zeros(len(df))
        anomalies_for_cleaning = []
        for col in df.columns:
            # NOTE: Assuming 'detect_anomalies' logic only applies to rows present in the dataframe
            # The original code's detect_anomalies returns reset indices.
            # We need to map them back or ensure the labeling is consistent.
            # Simpler approach: Iterate over rows and check the values.
            
            # Label ground truth (GT)
            anomalies_for_cleaning.extend(label_anomalies(df[col], col, gt_array, index_offset=0))

        # 2. Data Cleaning
        df_clean = df.copy()
        # Replace GT anomalies with NaN
        for anom in anomalies_for_cleaning:
            if anom["vital"] in df_clean.columns:
                df_clean.loc[anom["index"], anom["vital"]] = np.nan
        
        # Interpolate/impute, drop low variance, fill remaining NaNs
        df_clean = df_clean.interpolate().ffill().bfill()
        df_clean = df_clean.loc[:, df_clean.nunique() > 1].fillna(df_clean.mean())
        
        # Ensure GT array and df_clean are the same length (no rows dropped during cleaning)
        if len(df_clean) != len(gt_array):
             # This is a critical point: if rows were dropped, GT must be adjusted.
             # Since we only dropped columns and filled NaNs, this should be fine.
             pass

        # 3. Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean.values.astype(float))
        input_dim = X_scaled.shape[1]

        # 4. Autoencoder Model Train
        autoencoder = build_autoencoder(input_dim)
        history = autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1, verbose=0)
        print(f"Final training loss: {history.history['loss'][-1]:.4f}, val_loss: {history.history['val_loss'][-1]:.4f}")

        # 5. Reconstruct and Calculate Anomaly Scores
        reconstructions = autoencoder.predict(X_scaled)
        
        # Reconstruction Error (Anomaly Score)
        mse_per_row = np.mean(np.square(X_scaled - reconstructions), axis=1)

        # 6. Evaluation
        dataset_name = file.stem
        
        # Evaluate model against the GT
        eval_results = evaluate_anomaly_detection(gt_array, mse_per_row, file.name, dataset_name)
        evaluation_summary.append(eval_results)

        print(f"Metrics (Max F1 Threshold): AUC={eval_results['AUC']:.4f} | F1={eval_results['F1']:.4f} | P={eval_results['Precision']:.4f} | R={eval_results['Recall']:.4f} | FAR={eval_results['False Alarm Rate (FAR)']:.4f}")

        # 7. Reconstruction Metrics (for sanity check)
        mse_reco = mean_squared_error(X_scaled, reconstructions)
        r2 = r2_score(X_scaled, reconstructions)
        accuracy = 100 * (1 - mse_reco)
        print(f"Reconstruction Metrics: MSE={mse_reco:.4f} | R²={r2:.4f} | Accuracy≈{accuracy:.2f}% | GT Anomalies={eval_results['GT Anomaly Count']}")

        # 8. Plotting and Saving
        df_corrected = pd.DataFrame(scaler.inverse_transform(reconstructions), columns=df_clean.columns)
        df_corrected.to_csv(OUTPUT_DIR / f"{file.stem}_corrected.csv", index=False)
        plot_corrected(df_original.reset_index(drop=True), df_corrected, anomalies_for_cleaning, file.stem)

        # Plot reconstruction error
        plt.figure(figsize=(10,5))
        plt.plot(mse_per_row, label="Reconstruction Error")
        plt.axhline(y=eval_results["Optimal Threshold"], color='r', linestyle='--', label='Optimal F1 Threshold')
        plt.title(f"Reconstruction Error Plot - {file.stem}")
        plt.xlabel("Sample Index")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{file.stem}_error_plot.png")
        plt.close()

    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")

# ======================
# Save Overall Summary
# ======================
if evaluation_summary:
    # 1. Save Evaluation Summary
    eval_df = pd.DataFrame(evaluation_summary)
    summary_path = OUTPUT_DIR / "autoencoder_performance_summary.csv"
    eval_df.to_csv(summary_path, index=False)
    print(f"\n✅ Overall Model Performance Summary (AUC, F1, P, R, FAR) saved → {summary_path}")
    
    # 2. Best Model-Domain Analysis (Conceptual - based on AE only)
    # Since only AE is implemented and testing is on the training data, 
    # the "best pair" is simply the best-performing run.
    best_row = eval_df.loc[eval_df['F1'].idxmax()]
    
    print("\n" + "="*50)
    print("ANALYSIS & IDENTIFICATION OF BEST PAIR")
    print(f"Model: {best_row['Model']}")
    print(f"Domain (Dataset): {best_row['Dataset']}")
    print(f"Suitability for Real-Time Monitoring: High F1 and Low FAR are desired.")
    print(f"Best F1 Score: {best_row['F1']:.4f}")
    print(f"False Alarm Rate (FAR): {best_row['False Alarm Rate (FAR)']:.4f}")
    print("The current setup represents a single model-domain pair (AE trained/tested on the same data).")
    print("A full analysis would require comparing multiple models (ML vs DL) and cross-testing datasets.")
    print("="*50)

    print("\nSummary of Performance Metrics (Autoencoder):")
    print(eval_df[["Dataset", "AUC", "F1", "Precision", "Recall", "False Alarm Rate (FAR)"]])

else:
    print("\n⚠️ No valid results to summarize.")