
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("data/processed/binary_dataset.csv")
OUTPUT_FILE = Path("data/processed/binary_dataset_cleaned.csv")

CHUNK_SIZE = 200_000

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

total_rows = 0
total_infinities = 0
total_rows_affected = 0
total_missing_after = 0

print("=" * 70)
print("CLEANING NETWORK DATASET")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    # Identify numeric columns
    numeric_columns = chunk.select_dtypes(
        include=[np.number]
    ).columns

    # Count infinite values before cleaning
    infinity_mask = np.isinf(
        chunk[numeric_columns].to_numpy(dtype="float64")
    )

    infinity_count = int(infinity_mask.sum())

    affected_rows = int(infinity_mask.any(axis=1).sum())

    # Replace positive and negative infinity with NaN
    chunk[numeric_columns] = chunk[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Count missing values after cleaning
    missing_after = int(chunk.isna().sum().sum())

    total_rows += len(chunk)
    total_infinities += infinity_count
    total_rows_affected += affected_rows
    total_missing_after += missing_after

    chunk.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=(chunk_number == 1),
        index=False
    )

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows: {total_rows:,}"
    )

print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print("Total rows:", f"{total_rows:,}")
print("Infinite values replaced:", f"{total_infinities:,}")
print("Rows containing infinities:", f"{total_rows_affected:,}")
print("Missing cells remaining:", f"{total_missing_after:,}")
print("Output file:", OUTPUT_FILE)
print(
    "Output size (MB):",
    round(OUTPUT_FILE.stat().st_size / (1024 * 1024), 2)
)

print("\nOriginal dataset was not modified.")
