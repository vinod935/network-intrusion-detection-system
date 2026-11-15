
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "binary_dataset.csv"

CHUNK_SIZE = 100_000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in data/raw")

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

total_rows = 0
total_benign = 0
total_attack = 0

print("=" * 65)
print("CREATING COMBINED BINARY DATASET")
print("=" * 65)

for file in csv_files:

    print(f"\nProcessing: {file.name}")

    day = file.name.split("-")[0].title()

    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace"
    ):

        # Normalize column names
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            raise ValueError(f"Label column missing in {file.name}")

        # Normalize label whitespace
        chunk["Label"] = (
            chunk["Label"]
            .astype("string")
            .str.strip()
        )

        # Remove rows without a label
        chunk = chunk.dropna(subset=["Label"])

        # Binary target: BENIGN = 0, everything else = 1
        chunk["Target"] = (
            chunk["Label"].str.upper() != "BENIGN"
        ).astype("int8")

        # Preserve original source information
        chunk["Source_Day"] = day
        chunk["Source_File"] = file.name

        benign_count = int((chunk["Target"] == 0).sum())
        attack_count = int((chunk["Target"] == 1).sum())

        total_benign += benign_count
        total_attack += attack_count
        total_rows += len(chunk)

        # Write the chunk to disk
        chunk.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not OUTPUT_FILE.exists() or OUTPUT_FILE.stat().st_size == 0,
            index=False
        )

    print(f"Completed: {file.name}")

print("\n" + "=" * 65)
print("DATASET CREATION COMPLETED")
print("=" * 65)

print("Total rows:", f"{total_rows:,}")
print("Benign rows:", f"{total_benign:,}")
print("Attack rows:", f"{total_attack:,}")

print(
    "Benign percentage:",
    f"{total_benign / total_rows * 100:.2f}%"
)

print(
    "Attack percentage:",
    f"{total_attack / total_rows * 100:.2f}%"
)

print("Output file:", OUTPUT_FILE)
print("Output size (MB):", round(OUTPUT_FILE.stat().st_size / (1024 * 1024), 2))
