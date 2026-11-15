
import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/model_ready_dataset.csv")
CHUNK_SIZE = 200_000

metadata = ["Label", "Target", "Source_Day", "Source_File"]

total_rows = 0
target_counts = {0: 0, 1: 0}
missing_counts = {}
nonnumeric_counts = {}
first_values = {}
constant_features = set()
days = set()
source_files = set()

print("=" * 65)
print("VERIFYING MODEL-READY DATASET")
print("=" * 65)

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):

    total_rows += len(chunk)

    if total_rows == len(chunk):
        feature_columns = [c for c in chunk.columns if c not in metadata]

        for col in feature_columns:
            missing_counts[col] = 0
            nonnumeric_counts[col] = 0

    target_counts[0] += int((chunk["Target"] == 0).sum())
    target_counts[1] += int((chunk["Target"] == 1).sum())

    days.update(chunk["Source_Day"].dropna().unique())
    source_files.update(chunk["Source_File"].dropna().unique())

    for col in feature_columns:
        original = chunk[col]
        numeric = pd.to_numeric(original, errors="coerce")

        missing_counts[col] += int(original.isna().sum())
        nonnumeric_counts[col] += int(
            (original.notna() & numeric.isna()).sum()
        )

        valid = numeric.dropna()

        if len(valid) > 0:
            if col not in first_values:
                first_values[col] = valid.iloc[0]

            if not (valid == first_values[col]).all():
                constant_features.discard(col)
            elif col not in constant_features:
                # A feature is marked constant until a different value appears.
                pass

    print(f"Verified rows: {total_rows:,}")

# Recheck constant features accurately using a second chunked scan.
constant_features = set(feature_columns)

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    for col in feature_columns:
        values = pd.to_numeric(chunk[col], errors="coerce").dropna()

        if len(values) > 0 and values.nunique() > 1:
            constant_features.discard(col)

print("\n" + "=" * 65)
print("VERIFICATION RESULTS")
print("=" * 65)

print("Total rows:", f"{total_rows:,}")
print("Total columns:", len(feature_columns) + len(metadata))
print("Network features:", len(feature_columns))
print("Benign records:", f"{target_counts[0]:,}")
print("Attack records:", f"{target_counts[1]:,}")
print("Days:", sorted(days))
print("Source files:", len(source_files))

print("\nMissing values in network features:")
print(sum(missing_counts.values()))

print("Non-numeric values in network features:")
print(sum(nonnumeric_counts.values()))

print("Constant network features remaining:", len(constant_features))
if constant_features:
    print(sorted(constant_features))

print("\nTarget distribution:")
print(target_counts)

print("\nVerification completed.")
