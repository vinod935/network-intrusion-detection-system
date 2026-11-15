
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("data/processed/binary_dataset_sanitized.csv")
CHUNK_SIZE = 200_000

sanitized_columns = [
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

total_rows = 0
negative_counts = {col: 0 for col in sanitized_columns}
missing_counts = {col: 0 for col in sanitized_columns}
target_mismatches = 0

duplicate_equal_rows = 0
duplicate_different_rows = 0

forward_minus_one = 0
backward_minus_one = 0

print("=" * 70)
print("VERIFYING SANITIZED DATASET")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE),
    start=1
):
    total_rows += len(chunk)

    for col in sanitized_columns:
        negative_counts[col] += int((chunk[col] < 0).sum())
        missing_counts[col] += int(chunk[col].isna().sum())

    # Verify binary target against original label
    expected_target = (chunk["Label"] != "BENIGN").astype(int)
    target_mismatches += int(
        (chunk["Target"] != expected_target).sum()
    )

    # Check whether the two header-length columns are identical
    col_a = chunk["Fwd Header Length"]
    col_b = chunk["Fwd Header Length.1"]

    equal = col_a.eq(col_b) | (col_a.isna() & col_b.isna())

    duplicate_equal_rows += int(equal.sum())
    duplicate_different_rows += int((~equal).sum())

    # Check preserved -1 special values
    forward_minus_one += int(
        (chunk["Init_Win_bytes_forward"] == -1).sum()
    )
    backward_minus_one += int(
        (chunk["Init_Win_bytes_backward"] == -1).sum()
    )

    print(f"Verified rows: {total_rows:,}")

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"Total rows: {total_rows:,}")

print("\nNegative values remaining:")
for col, count in negative_counts.items():
    print(f"{col}: {count:,}")

print("\nMissing values after sanitization:")
for col, count in missing_counts.items():
    print(f"{col}: {count:,}")

print("\nTarget verification:")
print(f"Target mismatches: {target_mismatches:,}")

print("\nDuplicate feature investigation:")
print(f"Rows where Fwd Header Length columns match: {duplicate_equal_rows:,}")
print(f"Rows where they differ: {duplicate_different_rows:,}")

print("\nPreserved special values:")
print(f"Init_Win_bytes_forward = -1: {forward_minus_one:,}")
print(f"Init_Win_bytes_backward = -1: {backward_minus_one:,}")

print("\n" + "=" * 70)

if (
    total_rows == 2_830_743
    and sum(negative_counts.values()) == 0
    and target_mismatches == 0
):
    print("BASIC VERIFICATION PASSED")
else:
    print("VERIFICATION FAILED - INVESTIGATE THE RESULTS")

print("=" * 70)
