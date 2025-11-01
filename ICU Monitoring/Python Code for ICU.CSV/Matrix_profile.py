import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from stumpy import stump

print("\n=== Training Matrix Profile anomaly detection model on ICU.csv ===")

# =========================
# 1️⃣ Setup paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\Matrix-Profile"
os.makedirs(results_dir, exist_ok=True)

# =========================
# 2️⃣ Load and preprocess data
# =========================
df = pd.read_csv(input_path)
features = ['SysBP', 'Pulse']

df = df[features].dropna().reset_index(drop=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Analyze SysBP
signal = X_scaled[:, 0]

# =========================
# 3️⃣ Compute Matrix Profile
# =========================
window_size = 20
mp = stump(signal, m=window_size)
matrix_profile = mp[:, 0]

# =========================
# 4️⃣ Identify anomalies (discords)
# =========================
# The top matrix profile values are anomalies
num_anomalies = 5
discord_indices = np.argsort(matrix_profile)[-num_anomalies:][::-1]

# Mark anomalies
anomaly_flags = np.zeros(len(signal))
for idx in discord_indices:
    anomaly_flags[idx:idx+window_size] = 1

# =========================
# 5️⃣ Save results summary
# =========================
summary_text = (
    f"=== Matrix Profile Anomaly Detection Summary ===\n"
    f"Feature analyzed: SysBP\n"
    f"Window size: {window_size}\n"
    f"Total data points: {len(signal)}\n"
    f"Detected anomalies (top {num_anomalies} discords): {discord_indices.tolist()}\n"
)

print("\n" + summary_text)

# Save results
df_results = pd.DataFrame({
    "Index": np.arange(len(signal)),
    "SysBP": df["SysBP"],
    "Pulse": df["Pulse"],
    "MatrixProfile": np.append(matrix_profile, [np.nan] * (len(df) - len(matrix_profile))),
    "Anomaly": anomaly_flags.astype(int)
})

results_csv = os.path.join(results_dir, "MatrixProfile_Anomaly_Results.csv")
summary_txt = os.path.join(results_dir, "MatrixProfile_Anomaly_Summary.csv")
plot_path = os.path.join(results_dir, "MatrixProfile_Anomaly_Plot.png")

df_results.to_csv(results_csv, index=False)
with open(summary_txt, "w") as f:
    f.write(summary_text)

# =========================
# 6️⃣ Plot anomalies
# =========================
plt.figure(figsize=(12, 6))
plt.plot(df["SysBP"], label="SysBP", color='blue')
plt.scatter(
    np.where(anomaly_flags == 1),
    df["SysBP"][anomaly_flags == 1],
    color='red', label='Detected Anomaly', marker='x'
)
plt.title("Matrix Profile Anomaly Detection (SysBP)")
plt.xlabel("Index")
plt.ylabel("Systolic Blood Pressure (SysBP)")
plt.legend()
plt.grid(True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 7️⃣ Done
# =========================
print(f"✅ Matrix Profile anomaly detection completed successfully.")
print(f"   - Results: {results_csv}")
print(f"   - Summary: {summary_txt}")
print(f"   - Plot: {plot_path}")
