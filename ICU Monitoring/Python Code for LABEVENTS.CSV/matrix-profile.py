import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stumpy
from pathlib import Path
import chardet

# ===================== PATHS =====================
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\LABEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for LABEVENTS\MATRIX-PROFILE-LABEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of LABEVENTS"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== Processing {DATA_PATH.name} for Matrix Profile Anomaly Detection ===")

# ===================== ENCODING DETECTION =====================
with open(DATA_PATH, "rb") as f:
    rawdata = f.read(4096)
encoding = chardet.detect(rawdata)["encoding"]
print(f"✅ Detected encoding: {encoding}")

# ===================== DELIMITER DETECTION =====================
with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    sample_lines = [next(f) for _ in range(10)]
sample_text = "\n".join(sample_lines)
delims = [",", ";", "|", "\t"]
best_delim = max(delims, key=lambda d: sample_text.count(d))
print(f"🔍 Auto-detected best delimiter: '{best_delim}'")

# ===================== LOAD DATA =====================
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"✅ Loaded LABEVENTS.csv → Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ===================== PREPROCESSING =====================
# Convert time-related columns
for col in df.columns:
    if "time" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Create time difference if possible
if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
else:
    df["time_diff_min"] = np.arange(len(df))

# Use one representative numeric feature — e.g., 'valuenum' or 'value'
value_col = None
for col in ["valuenum", "value"]:
    if col in df.columns:
        value_col = col
        break
if not value_col:
    raise ValueError("❌ No numeric lab value column (e.g., 'valuenum' or 'value') found.")

# Drop NaN and invalids
df = df[[value_col, "time_diff_min"]].dropna()
df = df[df[value_col].apply(lambda x: isinstance(x, (int, float, np.number)))]

if df.empty:
    raise ValueError("❌ No valid numeric data found after cleaning.")

print(f"🧮 Using feature column: '{value_col}' | Total records: {len(df)}")

# ===================== MATRIX PROFILE =====================
window_size = max(10, len(df) // 100)  # Adaptive window size
print(f"\n🧠 Computing Matrix Profile with window size = {window_size}")

# Extract numeric series
series = df[value_col].astype(float).values

# Compute Matrix Profile
mp = stumpy.stump(series, m=window_size)
matrix_profile = mp[:, 0]

# Identify top anomalies (discords)
n_anomalies = min(10, len(series))
try:
    # Newer STUMPY versions (motifs.discords)
    discord_indices = stumpy.motifs.discords(matrix_profile, n_anomalies)
except Exception:
    # Manual fallback if motif.discords not found
    discord_indices = np.argsort(-matrix_profile)[:n_anomalies]

# If stumpy.motifs.discords returns a tuple
if isinstance(discord_indices, tuple):
    discord_indices = discord_indices[0]
discord_indices = np.array(discord_indices, dtype=int)

print(f"✅ Matrix Profile computed successfully.")
print(f"🚨 Top {len(discord_indices)} anomalies detected at indices: {discord_indices.tolist()}")

# ===================== SAVE RESULTS =====================
# Pad matrix profile to match series length
if len(matrix_profile) < len(series):
    pad_len = len(series) - len(matrix_profile)
    matrix_profile = np.pad(matrix_profile, (0, pad_len), constant_values=np.nan)

# Mark anomalies
is_anomaly = np.zeros(len(series), dtype=int)
is_anomaly[discord_indices] = 1

results_df = pd.DataFrame({
    "index": np.arange(len(series)),
    value_col: series,
    "matrix_profile": matrix_profile,
    "is_anomaly": is_anomaly
})

results_file = OUTPUT_DIR / "LABEVENTS_MatrixProfile_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"✅ Results saved → {results_file}")

summary = {
    "total_records": len(series),
    "window_size": window_size,
    "top_anomalies_detected": len(discord_indices),
    "discord_indices": discord_indices.tolist(),
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "MatrixProfile_LABEVENTS_Summary.csv", index=False)
print(f"✅ Summary saved → {SUMMARY_DIR / 'MatrixProfile_LABEVENTS_Summary.csv'}")

# ===================== VISUALIZE =====================
plt.figure(figsize=(12, 6))
plt.plot(series, label=f"{value_col} values", alpha=0.6)
plt.scatter(discord_indices, series[discord_indices], color="red", marker="x", label="Detected Anomalies (Discords)")
plt.title("Matrix Profile - Anomaly Detection (LABEVENTS)")
plt.xlabel("Index")
plt.ylabel(value_col)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "LABEVENTS_MatrixProfile_AnomalyPlot.png")
plt.close()
print(f"✅ Plot saved → {OUTPUT_DIR / 'LABEVENTS_MatrixProfile_AnomalyPlot.png'}")

print("\n🎯 Matrix Profile–based Anomaly Detection completed successfully!")
