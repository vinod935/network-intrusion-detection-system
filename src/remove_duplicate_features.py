
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/binary_dataset_sanitized.csv")
OUTPUT_FILE = Path("data/processed/binary_dataset_final.csv")

CHUNK_SIZE = 200_000
DUPLICATE_COLUMN = "Fwd Header Length.1"

total_rows = 0

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

print("=" * 70)
print("REMOVING VERIFIED DUPLICATE FEATURE")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE),
    start=1
):
    if DUPLICATE_COLUMN not in chunk.columns:
        raise ValueError("Duplicate column not found.")

    chunk.drop(columns=[DUPLICATE_COLUMN], inplace=True)

    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if chunk_number == 1 else "a",
        header=(chunk_number == 1),
        index=False
    )

    total_rows += len(chunk)

    print(f"Processed rows: {total_rows:,}")

print("\n" + "=" * 70)
print("DUPLICATE FEATURE REMOVAL COMPLETED")
print("=" * 70)

print(f"Total rows: {total_rows:,}")
print(f"Removed feature: {DUPLICATE_COLUMN}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Final columns: {len(chunk.columns)}")
print(
    "Output size (MB):",
    round(OUTPUT_FILE.stat().st_size / (1024 * 1024), 2)
)
