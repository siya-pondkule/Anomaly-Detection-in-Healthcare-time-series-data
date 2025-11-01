import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stumpy
from pathlib import Path
import csv

# ========== Paths ==========
DATA_PATH = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Datasets\DATETIMEEVENTS.csv")
OUTPUT_DIR = Path(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models for DATETIMEEVENTS\MatrixProfile-DATETIMEEVENTS")
SUMMARY_DIR = OUTPUT_DIR.parent / "Summary of DATETIMEEVENTS"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
SUMMARY_DIR.mkdir(exist_ok=True, parents=True)

print(f"\n=== Processing {DATA_PATH.name} for Matrix Profile–based Anomaly Detection ===")

# ========== Encoding & delimiter detection ==========
try:
    with open(DATA_PATH, "rb") as f:
        sample = f.read(4096)
    encoding = "utf-8"
    sample.decode(encoding)
except Exception:
    encoding = "latin1"
print(f"Detected encoding: {encoding}")

# Detect delimiter
with open(DATA_PATH, "r", encoding=encoding, errors="ignore") as f:
    lines = [next(f) for _ in range(5)]
sample_text = "\n".join(lines)
delim_candidates = [",", ";", "|", "\t"]
delim_counts = {d: sample_text.count(d) for d in delim_candidates}
best_delim = max(delim_counts, key=delim_counts.get)
print(f"Detected delimiter: '{best_delim}'")

# ========== Load CSV ==========
df = pd.read_csv(DATA_PATH, encoding=encoding, delimiter=best_delim, engine="python", on_bad_lines="skip")
print(f"Loaded data shape: {df.shape}")

# ========== Preprocessing ==========
if "charttime" in df.columns:
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

# Create a time_diff feature if not numeric
if "subject_id" in df.columns and "charttime" in df.columns:
    df.sort_values(["subject_id", "charttime"], inplace=True)
    df["time_diff_min"] = df.groupby("subject_id")["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
elif "charttime" in df.columns:
    df.sort_values("charttime", inplace=True)
    df["time_diff_min"] = df["charttime"].diff().dt.total_seconds() / 60.0
    df["time_diff_min"] = df["time_diff_min"].fillna(0.0)
else:
    df["time_diff_min"] = np.arange(len(df)).astype(float)

# Choose numeric column (prefer 'value' or 'time_diff_min')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_col = None
for c in ["value", "valuenum", "time_diff_min"]:
    if c in numeric_cols:
        selected_col = c
        break
if selected_col is None:
    selected_col = numeric_cols[0]

print(f"Using column for Matrix Profile: {selected_col}")
series = df[selected_col].astype(float).values

# ========== Compute Matrix Profile ==========
WINDOW = 50  # subsequence length, tuneable
if len(series) < WINDOW * 2:
    raise ValueError(f"Series too short ({len(series)} points) for window {WINDOW}")

# Compute matrix profile (self-join)
print("Computing matrix profile...")
mp = stumpy.stump(series, m=WINDOW)
matrix_profile = mp[:, 0]

# Identify top discord (most anomalous subsequence)
discord_idx = np.argmax(matrix_profile)
discord_score = matrix_profile[discord_idx]
print(f"Top anomaly subsequence starts at index {discord_idx} with discord score {discord_score:.3f}")

# ========== Save Results ==========
results_df = pd.DataFrame({
    "index": np.arange(len(matrix_profile)),
    "matrix_profile_value": matrix_profile
})
results_df["is_anomaly"] = 0
results_df.loc[discord_idx, "is_anomaly"] = 1

results_file = OUTPUT_DIR / "DATETIMEEVENTS_MatrixProfile_Results.csv"
results_df.to_csv(results_file, index=False)
print(f"Results saved → {results_file}")

summary = {
    "series_length": len(series),
    "window_size": WINDOW,
    "top_anomaly_index": int(discord_idx),
    "discord_score": float(discord_score)
}
pd.DataFrame([summary]).to_csv(SUMMARY_DIR / "MatrixProfile_DATETIMEEVENTS_Summary.csv", index=False)
print(f"Summary saved → {SUMMARY_DIR / 'MatrixProfile_DATETIMEEVENTS_Summary.csv'}")

# ========== Visualization ==========
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(series, label="Time series")
plt.axvline(discord_idx, color="red", linestyle="--", label="Anomaly start")
plt.legend()
plt.title(f"DATETIMEEVENTS Time Series ({selected_col})")

plt.subplot(2, 1, 2)
plt.plot(matrix_profile, label="Matrix Profile")
plt.axvline(discord_idx, color="red", linestyle="--", label="Detected Anomaly")
plt.xlabel("Index")
plt.ylabel("Profile Value")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "DATETIMEEVENTS_MatrixProfile_AnomalyPlot.png")
plt.close()

print(f"Plot saved → {OUTPUT_DIR / 'DATETIMEEVENTS_MatrixProfile_AnomalyPlot.png'}")
print("\n✅ Matrix Profile anomaly detection finished successfully.")
