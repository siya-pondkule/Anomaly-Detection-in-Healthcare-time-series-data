import pandas as pd

df = pd.read_csv(r"D:\Final Year\Project\Anomaly Detection\Anomaly Detection\ICU Monitoring\ICU Datasets\ICU.csv")
print("\n=== Columns in ICU.csv ===")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
