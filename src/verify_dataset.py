
import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/processed/binary_dataset.csv")
CHUNK_SIZE = 200_000

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

total_rows = 0
total_benign = 0
total_attack = 0
target_mismatches = 0
missing_values = 0

day_counts = {}
file_counts = {}

print("=" * 70)
print("COMBINED DATASET VERIFICATION")
print("=" * 70)

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=["Label", "Target", "Source_Day", "Source_File"],
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    chunk["Label"] = chunk["Label"].astype("string").str.strip()

    # Check whether Target agrees with the original Label
    expected_target = (
        chunk["Label"].str.upper() != "BENIGN"
    ).astype("int8")

    target_mismatches += int(
        (chunk["Target"] != expected_target).sum()
    )

    total_rows += len(chunk)
    total_benign += int((chunk["Target"] == 0).sum())
    total_attack += int((chunk["Target"] == 1).sum())

    missing_values += int(
        chunk[["Label", "Target", "Source_Day", "Source_File"]]
        .isna()
        .sum()
        .sum()
    )

    # Count flows by day
    for day, count in chunk["Source_Day"].value_counts().items():
        day_counts[day] = day_counts.get(day, 0) + int(count)

    # Count flows by original CSV file
    for file, count in chunk["Source_File"].value_counts().items():
        file_counts[file] = file_counts.get(file, 0) + int(count)

print("\nOVERALL CHECK")
print("Total rows:", f"{total_rows:,}")
print("Benign:", f"{total_benign:,}")
print("Attack:", f"{total_attack:,}")
print("Target-label mismatches:", target_mismatches)
print("Missing values in verification columns:", missing_values)

print("\nDAY-WISE DISTRIBUTION")

for day, count in sorted(day_counts.items()):
    print(f"{day:12s} {count:12,}")

print("\nSOURCE FILE DISTRIBUTION")

for file, count in sorted(file_counts.items()):
    print(f"{file}: {count:,}")

print("\nVERIFICATION RESULT")
print("Unique days:", len(day_counts))
print("Unique source files:", len(file_counts))

if (
    total_rows == 2_830_743
    and total_benign == 2_273_097
    and total_attack == 557_646
    and target_mismatches == 0
    and missing_values == 0
    and len(file_counts) == 8
):
    print("\nSUCCESS: Combined dataset passed the verification checks.")
else:
    print("\nWARNING: One or more verification checks need investigation.")
