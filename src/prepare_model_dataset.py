
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/binary_dataset_final.csv")
OUTPUT_FILE = Path("data/processed/model_ready_dataset.csv")

CHUNK_SIZE = 200_000

# Previously verified constant features
CONSTANT_FEATURES = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate"
]

METADATA_COLUMNS = [
    "Label",
    "Target",
    "Source_Day",
    "Source_File"
]

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

total_rows = 0
expected_features = None

print("=" * 70)
print("PREPARING MODEL-READY DATASET")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE),
    start=1
):
    missing = [
        col for col in CONSTANT_FEATURES + METADATA_COLUMNS
        if col not in chunk.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Remove only the eight verified constant features.
    chunk = chunk.drop(columns=CONSTANT_FEATURES)

    # Verify the remaining network features are numeric.
    feature_columns = [
        col for col in chunk.columns
        if col not in METADATA_COLUMNS
    ]

    non_numeric = [
        col for col in feature_columns
        if not pd.api.types.is_numeric_dtype(chunk[col])
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric network features: {non_numeric}"
        )

    if expected_features is None:
        expected_features = len(feature_columns)

    total_rows += len(chunk)

    chunk.to_csv(
        OUTPUT_FILE,
        mode="w" if chunk_number == 1 else "a",
        header=(chunk_number == 1),
        index=False
    )

    print(f"Processed rows: {total_rows:,}")

print("\n" + "=" * 70)
print("PREPARATION COMPLETED")
print("=" * 70)

print(f"Total rows: {total_rows:,}")
print(f"Network features: {expected_features}")
print(f"Removed constant features: {len(CONSTANT_FEATURES)}")
print(f"Output: {OUTPUT_FILE}")
print(
    "File size (MB):",
    round(OUTPUT_FILE.stat().st_size / (1024 * 1024), 2)
)
