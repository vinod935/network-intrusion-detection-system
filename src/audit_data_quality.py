
import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path("data/processed/binary_dataset.csv")
CHUNK_SIZE = 200_000

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

metadata_columns = {
    "Label",
    "Target",
    "Source_Day",
    "Source_File"
}

audit = {}
total_rows = 0

print("=" * 75)
print("NETWORK FEATURE DATA QUALITY AUDIT")
print("=" * 75)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    chunk.columns = chunk.columns.str.strip()

    feature_columns = [
        col for col in chunk.columns
        if col not in metadata_columns
    ]

    for col in feature_columns:

        if col not in audit:
            audit[col] = {
                "missing": 0,
                "infinite": 0,
                "non_numeric": 0
            }

        original = chunk[col]

        # Count existing missing values
        audit[col]["missing"] += int(original.isna().sum())

        # Convert feature values to numeric
        numeric = pd.to_numeric(original, errors="coerce")

        # Count unexpected non-numeric values
        invalid = numeric.isna() & original.notna()
        audit[col]["non_numeric"] += int(invalid.sum())

        # Count positive and negative infinity
        values = numeric.to_numpy(
            dtype="float64",
            na_value=np.nan
        )

        audit[col]["infinite"] += int(
            np.isinf(values).sum()
        )

    total_rows += len(chunk)

    print(
        f"Processed chunk {chunk_number} | "
        f"Total rows: {total_rows:,}"
    )

print("\n" + "=" * 75)
print("AUDIT RESULTS")
print("=" * 75)

total_missing = 0
total_infinite = 0
total_non_numeric = 0

affected_columns = 0

for col, counts in audit.items():

    missing = counts["missing"]
    infinite = counts["infinite"]
    non_numeric = counts["non_numeric"]

    total_missing += missing
    total_infinite += infinite
    total_non_numeric += non_numeric

    if missing > 0 or infinite > 0 or non_numeric > 0:
        affected_columns += 1

        print(f"\nFeature: {col}")
        print(f"  Missing values: {missing:,}")
        print(f"  Infinite values: {infinite:,}")
        print(f"  Non-numeric values: {non_numeric:,}")

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print("Total rows scanned:", f"{total_rows:,}")
print("Total network features:", len(audit))
print("Features with data-quality issues:", affected_columns)
print("Total missing cells:", f"{total_missing:,}")
print("Total infinite cells:", f"{total_infinite:,}")
print("Total non-numeric cells:", f"{total_non_numeric:,}")

if affected_columns == 0:
    print("\nAll network features passed these checks.")
else:
    print("\nData-quality issues detected. Review the feature-level results.")

print("\nAudit completed. Original dataset was not modified.")
