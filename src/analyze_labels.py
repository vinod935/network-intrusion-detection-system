
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
CHUNK_SIZE = 200_000

csv_files = sorted(DATA_DIR.glob("*.csv"))

total_rows = 0
overall_labels = {}

print("=" * 75)
print("CIC-IDS2017 FULL DATASET LABEL ANALYSIS")
print("=" * 75)

for file in csv_files:
    print(f"\nProcessing: {file.name}")

    file_rows = 0
    file_labels = {}

    try:
        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):
            chunk.columns = chunk.columns.str.strip()

            if "Label" not in chunk.columns:
                raise ValueError("Label column not found!")

            labels = chunk["Label"].astype("string").str.strip()
            counts = labels.value_counts(dropna=False)

            for label, count in counts.items():
                label = "<MISSING>" if pd.isna(label) else label

                file_labels[label] = (
                    file_labels.get(label, 0) + int(count)
                )

                overall_labels[label] = (
                    overall_labels.get(label, 0) + int(count)
                )

            file_rows += len(chunk)

        total_rows += file_rows

        print("Total rows:", f"{file_rows:,}")
        print("Attack categories:", len(file_labels))

        for label, count in sorted(file_labels.items()):
            print(f"  {label}: {count:,}")

    except Exception as e:
        print("ERROR:", e)
        raise

print("\n" + "=" * 75)
print("OVERALL DATASET SUMMARY")
print("=" * 75)

print("Total CSV files:", len(csv_files))
print("Total rows:", f"{total_rows:,}")
print("Total categories:", len(overall_labels))

print("\nOverall label distribution:")

for label, count in sorted(overall_labels.items()):
    percentage = (count / total_rows) * 100 if total_rows else 0

    print(
        f"{label:25s} {count:12,} "
        f"({percentage:.4f}%)"
    )

print("\nLabel analysis completed!")
