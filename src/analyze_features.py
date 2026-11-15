
import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path("data/processed/binary_dataset_cleaned.csv")
CHUNK_SIZE = 200_000

metadata_columns = {
    "Label",
    "Target",
    "Source_Day",
    "Source_File"
}

if not DATA_FILE.exists():
    raise FileNotFoundError(DATA_FILE)

feature_stats = {}
total_rows = 0

print("=" * 70)
print("FEATURE ANALYSIS")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    total_rows += len(chunk)

    features = [
        col for col in chunk.columns
        if col not in metadata_columns
    ]

    for col in features:

        values = pd.to_numeric(chunk[col], errors="coerce")

        if col not in feature_stats:
            feature_stats[col] = {
                "min": np.inf,
                "max": -np.inf,
                "missing": 0,
                "valid": 0
            }

        stats = feature_stats[col]

        stats["missing"] += int(values.isna().sum())

        valid = values.dropna()

        if len(valid) > 0:
            stats["min"] = min(stats["min"], float(valid.min()))
            stats["max"] = max(stats["max"], float(valid.max()))
            stats["valid"] += len(valid)

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows: {total_rows:,}"
    )

constant_features = []
problem_features = []

print("\n" + "=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

for col, stats in feature_stats.items():

    minimum = stats["min"]
    maximum = stats["max"]

    if stats["valid"] == 0:
        status = "NO VALID VALUES"
        problem_features.append(col)

    elif minimum == maximum:
        status = "CONSTANT"
        constant_features.append(col)

    else:
        status = "VARIABLE"

    print(f"\n{col}")
    print(f"  Minimum: {minimum}")
    print(f"  Maximum: {maximum}")
    print(f"  Missing values: {stats['missing']:,}")
    print(f"  Status: {status}")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("Total rows:", f"{total_rows:,}")
print("Total features:", len(feature_stats))
print("Constant features:", len(constant_features))
print("Features with no valid values:", len(problem_features))

print("\nConstant feature names:")
for col in constant_features:
    print(" -", col)

print("\nFeature analysis completed.")
