import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

print("\n=== Training One-Class SVM anomaly detection model on ICU.csv ===")

# =========================
# 1️⃣ Setup paths
# =========================
input_path = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv"
results_dir = r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\Results of all Models\One-Class-SVM"
os.makedirs(results_dir, exist_ok=True)

# =========================
# 2️⃣ Load and preprocess data
# =========================
df = pd.read_csv(input_path)

# Use same columns as Isolation Forest / LOF
features = ['Age', 'SysBP', 'Pulse']
df = df[features].dropna().reset_index(drop=True)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================
# 3️⃣ Train One-Class SVM
# =========================
# nu → expected anomaly proportion, gamma → kernel coefficient
svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
y_pred = svm.fit_predict(X_scaled)

# Convert prediction: -1 = anomaly, 1 = normal
df['Anomaly'] = np.where(y_pred == -1, 1, 0)

# =========================
# 4️⃣ Summary
# =========================
num_anomalies = df['Anomaly'].sum()
summary_text = (
    f"=== One-Class SVM Anomaly Detection Summary ===\n"
    f"Total samples: {len(df)}\n"
    f"Detected anomalies: {num_anomalies}\n"
    f"Normal points: {len(df) - num_anomalies}\n"
    f"Features used: {features}\n"
    f"Kernel: RBF | nu=0.05 | gamma='scale'\n"
)

print("\n" + summary_text)

# =========================
# 5️⃣ Save results
# =========================
results_csv = os.path.join(results_dir, "OneClassSVM_Anomaly_Results.csv")
summary_txt = os.path.join(results_dir, "OneClassSVM_Anomaly_Summary.csv")
plot_path = os.path.join(results_dir, "OneClassSVM_Anomaly_Plot.png")

df.to_csv(results_csv, index=False)
with open(summary_txt, "w") as f:
    f.write(summary_text)

# =========================
# 6️⃣ Visualization
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(df['SysBP'], df['Pulse'], c=df['Anomaly'], cmap='coolwarm', edgecolors='k')
plt.xlabel("Systolic Blood Pressure (SysBP)")
plt.ylabel("Pulse")
plt.title("One-Class SVM Anomaly Detection on ICU Data")
plt.grid(True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# =========================
# 7️⃣ Done
# =========================
print(f"✅ One-Class SVM anomaly detection completed successfully.")
print(f"   - Results: {results_csv}")
print(f"   - Summary: {summary_txt}")
print(f"   - Plot: {plot_path}")
