import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

print("\n=== Training LOF anomaly detection model on ICU.csv ===")

# =========================
# 1️⃣ Setup paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\LOF-ModelResult"

# Create results directory if it doesn’t exist
os.makedirs(results_dir, exist_ok=True)

# =========================
# 2️⃣ Load dataset
# =========================
df = pd.read_csv(input_path)

# Select meaningful numerical columns for anomaly detection
features = ['Age', 'SysBP', 'Pulse']

# Drop missing values and reset index
df = df[features].dropna().reset_index(drop=True)

# =========================
# 3️⃣ Preprocessing
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================
# 4️⃣ Train LOF model
# =========================
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_pred = lof.fit_predict(X_scaled)

# Convert prediction (-1 = anomaly, 1 = normal)
df['Anomaly'] = np.where(y_pred == -1, 1, 0)

# =========================
# 5️⃣ Summary
# =========================
num_anomalies = df['Anomaly'].sum()
summary_text = (
    f"=== LOF Anomaly Detection Summary ===\n"
    f"Total samples: {len(df)}\n"
    f"Detected anomalies: {num_anomalies}\n"
    f"Normal points: {len(df) - num_anomalies}\n"
    f"Contamination: 5%\n"
)

print("\n" + summary_text)

# =========================
# 6️⃣ Save results
# =========================
results_csv = os.path.join(results_dir, "LOF_Anomaly_Results.csv")
summary_txt = os.path.join(results_dir, "LOF_Anomaly_Summary.txt")
plot_path = os.path.join(results_dir, "LOF_Anomaly_Plot.png")

df.to_csv(results_csv, index=False)

with open(summary_txt, "w") as f:
    f.write(summary_text)

# =========================
# 7️⃣ Plot and save figure
# =========================
plt.figure(figsize=(8,6))
plt.scatter(df['SysBP'], df['Pulse'], c=df['Anomaly'], cmap='coolwarm')
plt.xlabel("Systolic Blood Pressure (SysBP)")
plt.ylabel("Pulse")
plt.title("LOF Anomaly Detection on ICU Data")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 8️⃣ Final message
# =========================
print(f"✅ Results saved successfully in: {results_dir}")
print(f"   - CSV: {results_csv}")
print(f"   - Summary: {summary_txt}")
print(f"   - Plot: {plot_path}")
