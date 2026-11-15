
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

stats = {}
total_rows = 0

print("=" * 70)
print("NEGATIVE VALUE INVESTIGATION")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    features = [
        col for col in chunk.columns
        if col not in metadata_columns
    ]

    for col in features:

        values = pd.to_numeric(chunk[col], errors="coerce")

        if col not in stats:
            stats[col] = {
                "negative": 0,
                "zero": 0,
                "positive": 0,
                "minimum": np.inf,
                "maximum": -np.inf
            }

        valid = values.dropna()

        stats[col]["negative"] += int((valid < 0).sum())
        stats[col]["zero"] += int((valid == 0).sum())
        stats[col]["positive"] += int((valid > 0).sum())

        if len(valid):
            stats[col]["minimum"] = min(
                stats[col]["minimum"], float(valid.min())
            )
            stats[col]["maximum"] = max(
                stats[col]["maximum"], float(valid.max())
            )

    total_rows += len(chunk)
    print(f"Processed chunk {chunk_number}: {total_rows:,} rows")

print("\n" + "=" * 70)
print("FEATURES CONTAINING NEGATIVE VALUES")
print("=" * 70)

negative_features = []

for col, values in stats.items():

    if values["negative"] > 0:
        negative_features.append(col)

        print(f"\n{col}")
        print(f"  Negative values: {values['negative']:,}")
        print(f"  Zero values: {values['zero']:,}")
        print(f"  Positive values: {values['positive']:,}")
        print(f"  Minimum: {values['minimum']}")
        print(f"  Maximum: {values['maximum']}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("Total rows:", f"{total_rows:,}")
print("Features containing negative values:", len(negative_features))

print("\nInvestigation completed. No data was modified.")
