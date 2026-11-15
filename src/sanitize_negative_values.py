
import pandas as pd
from pathlib import Path
import os

INPUT_FILE = Path("data/processed/binary_dataset_cleaned.csv")
OUTPUT_FILE = Path("data/processed/binary_dataset_sanitized.csv")
TEMP_FILE = Path("data/processed/binary_dataset_sanitized_temp.csv")

CHUNK_SIZE = 200_000
EXPECTED_ROWS = 2_830_743

INVALID_NEGATIVE_FEATURES = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Header Length.1",
    "min_seg_size_forward"
]

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

# Remove only an old temporary file, not the existing final dataset.
if TEMP_FILE.exists():
    TEMP_FILE.unlink()

total_rows = 0
total_replaced = 0

replacement_counts = {
    feature: 0 for feature in INVALID_NEGATIVE_FEATURES
}

print("=" * 70)
print("SAFE DATASET SANITIZATION")
print("=" * 70)

try:
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ),
        start=1
    ):

        missing_columns = [
            col for col in INVALID_NEGATIVE_FEATURES
            if col not in chunk.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        for feature in INVALID_NEGATIVE_FEATURES:
            values = pd.to_numeric(
                chunk[feature],
                errors="coerce"
            )

            negative_mask = values < 0
            count = int(negative_mask.sum())

            replacement_counts[feature] += count
            total_replaced += count

            chunk.loc[negative_mask, feature] = float("nan")

        total_rows += len(chunk)

        chunk.to_csv(
            TEMP_FILE,
            mode="w" if chunk_number == 1 else "a",
            header=(chunk_number == 1),
            index=False
        )

        print(
            f"Chunk {chunk_number} completed | "
            f"Rows: {total_rows:,}"
        )

    print("\nChecking total row count...")

    if total_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} rows, "
            f"but processed {total_rows:,}. "
            "Final dataset was not replaced."
        )

    if not TEMP_FILE.exists() or TEMP_FILE.stat().st_size == 0:
        raise ValueError("Temporary output file is missing or empty.")

    # Replace final file only after successful processing.
    os.replace(TEMP_FILE, OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("SANITIZATION COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"Total rows: {total_rows:,}")
    print(f"Invalid negative values replaced: {total_replaced:,}")

    print("\nReplacements by feature:")
    for feature, count in replacement_counts.items():
        print(f"{feature}: {count:,}")

    print("\nSpecial values preserved:")
    print("Init_Win_bytes_forward = -1")
    print("Init_Win_bytes_backward = -1")

    print("\nFinal output:", OUTPUT_FILE)
    print(
        "File size (MB):",
        round(OUTPUT_FILE.stat().st_size / (1024 * 1024), 2)
    )

except Exception:
    print("\nERROR: Sanitization did not complete.")
    print("The previous final dataset has not been replaced.")
    raise
