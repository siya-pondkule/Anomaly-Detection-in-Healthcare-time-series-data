import os
import zipfile
import pandas as pd
import wfdb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import shutil


def preprocess_tabular_dataset(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def preprocess_wfdb_record(record_path, output_dir):
    record_name = os.path.basename(record_path)

    # Check if both .hea and .dat files exist
    if not (os.path.exists(record_path + ".hea") and os.path.exists(record_path + ".dat")):
        print(f"⚠️ Skipping incomplete WFDB record (missing .hea or .dat): {record_name}")
        return

    try:
        rec = wfdb.rdrecord(record_path)
        try:
            ann = wfdb.rdann(record_path, 'atr')
        except Exception:
            ann = None

        df = pd.DataFrame(rec.p_signal, columns=rec.sig_name)
        df['sample_index'] = range(len(df))

        if ann is not None:
            ann_df = pd.DataFrame({
                'annotation_sample': ann.sample,
                'annotation_symbol': ann.symbol
            })
            df = df.merge(ann_df, how='left', left_on='sample_index', right_on='annotation_sample')
            df.drop(columns=['annotation_sample'], inplace=True)

        output_csv = os.path.join(output_dir, f"{record_name}.csv")
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved processed WFDB record (includes .hea/.dat/.atr/.xws): {output_csv}")

    except Exception as e:
        print(f"❌ Error processing WFDB record {record_name}: {e}")


def process_zip(zip_path, output_dir="processed_datasets/Dataset2"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    temp_dir = "temp_data"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    print(f"\n📦 Extracted ZIP: {zip_path}\n")

    wfdb_records = set()
    for root, _, files in os.walk(temp_dir):
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in ['.dat', '.hea', '.atr', '.xws']:
                wfdb_records.add(os.path.join(root, base))

    processed = set()

    for record_path in wfdb_records:
        if record_path not in processed:
            preprocess_wfdb_record(record_path, output_dir)
            processed.add(record_path)

    for root, _, files in os.walk(temp_dir):
        for f in files:
            file_path = os.path.join(root, f)
            ext = f.lower().split('.')[-1]

            if ext in ['csv']:
                df = pd.read_csv(file_path)
                df = preprocess_tabular_dataset(df)
                df.to_csv(os.path.join(output_dir, f"processed_{f}"), index=False)
                print(f"✅ Processed CSV file: {f}")

            elif ext in ['xls', 'xlsx']:
                df = pd.read_excel(file_path)
                df = preprocess_tabular_dataset(df)
                df.to_csv(os.path.join(output_dir, f"processed_{f}.csv"), index=False)
                print(f"✅ Processed Excel file: {f}")

    print("\n🎉 All WFDB + Tabular files processed successfully!")

    try:
        shutil.rmtree(temp_dir)
        print("🧹 Temporary files cleaned up.")
    except Exception as e:
        print(f"⚠️ Could not delete temp folder: {e}")


if __name__ == "__main__":
    process_zip("./Datasets/mit-bih-normal-sinus-rhythm-database-1.0.0.zip")
