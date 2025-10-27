import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

print("Script started")

data_dir = Path(r"d:/Final Year/Project/Anomaly Detection/ICU Monitoring/Datasets")
print("Folder exists:", data_dir.exists())
print("Absolute path:", data_dir.resolve())

if not data_dir.exists():
    print("Folder does not exist!")
else:
    # List all files and folders recursively
    all_files = list(data_dir.rglob("*"))
    print(f"\nAll files and folders under {data_dir}:")
    for f in all_files:
        print(f" {f}" if f.is_dir() else f" {f} (extension: {f.suffix})")

    # Detect CSV files (case-insensitive)
    csv_files = [f for f in all_files if f.is_file() and f.suffix.lower() == ".csv"]

    if not csv_files:
        print(" No CSV files found in the folder or its subfolders.")
    else:
        print(f"\nFound CSV files: {[f.name for f in csv_files]}")

        # Create a folder to save plots
        output_dir = data_dir / "plots"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nPlots will be saved to: {output_dir}")

        # Iterate over all CSV files
        for file_path in csv_files:
            filename = file_path.name
            print(f"\n🔹 Processing {filename}...")

            try:
                # Read CSV (limit rows for large files)
                df = pd.read_csv(file_path, nrows=50000)
                print(f"Columns in CSV: {df.columns.tolist()}")

                # Detect datetime and numeric columns
                datetime_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                print(f"Datetime columns found: {datetime_cols}")
                print(f"Numeric columns found: {numeric_cols}")

                if datetime_cols and numeric_cols:
                    df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]], errors='coerce')
                    df = df.dropna(subset=[datetime_cols[0]])
                    df = df.sort_values(by=datetime_cols[0])


                    # Plot all numeric columns vs datetime
                    plt.figure(figsize=(10, 5))
                    for col in numeric_cols:
                        plt.plot(df[datetime_cols[0]], df[col], label=col, alpha=0.7)

                    plt.xlabel(datetime_cols[0])
                    plt.ylabel("Values")
                    plt.title(f"Time Series Plot - {filename}")
                    plt.legend(loc='best', fontsize='small')
                    plt.tight_layout()

                    # Save plot
                    plot_path = output_dir / f"{filename}_plot.png"
                    plt.savefig(plot_path, dpi=150)
                    print(f"Plot saved: {plot_path}")
                    plt.show()
                    plt.close()
                else:
                    print("Skipped (no datetime/numeric columns found).")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

print("\nScript finished")
