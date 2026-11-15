
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("No CSV files found!")
    exit()

print("=" * 70)
print("CIC-IDS2017 DATASET INSPECTION")
print("=" * 70)

for file in csv_files:
    print(f"\nFILE: {file.name}")
    print("-" * 70)

    try:
        df = pd.read_csv(file, nrows=5, low_memory=False)

        # Remove accidental spaces from column names for inspection
        df.columns = df.columns.str.strip()

        print("Number of columns:", len(df.columns))
        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes)

        label_column = next(
            (col for col in df.columns if col.lower() == "label"),
            None
        )

        if label_column:
            print("\nSample labels:")
            print(df[label_column].value_counts())
        else:
            print("\nWARNING: Label column not found!")

        print("\nFirst 5 records:")
        print(df.head())

    except Exception as e:
        print("Error:", e)

print("\nInspection completed!")
