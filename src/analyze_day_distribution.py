
import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/processed/binary_dataset_cleaned.csv")
CHUNK_SIZE = 200_000

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

day_stats = {}
file_stats = {}

print("=" * 70)
print("ATTACK DISTRIBUTION BY DAY AND SOURCE FILE")
print("=" * 70)

for chunk_number, chunk in enumerate(
    pd.read_csv(
        DATA_FILE,
        usecols=["Target", "Source_Day", "Source_File"],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    for day, group in chunk.groupby("Source_Day"):
        stats = day_stats.setdefault(
            day, {"benign": 0, "attack": 0}
        )

        stats["benign"] += int((group["Target"] == 0).sum())
        stats["attack"] += int((group["Target"] == 1).sum())

    for file, group in chunk.groupby("Source_File"):
        stats = file_stats.setdefault(
            file, {"benign": 0, "attack": 0}
        )

        stats["benign"] += int((group["Target"] == 0).sum())
        stats["attack"] += int((group["Target"] == 1).sum())

    print(f"Processed chunk {chunk_number}")

print("\n" + "=" * 70)
print("DAY-WISE DISTRIBUTION")
print("=" * 70)

for day, stats in sorted(day_stats.items()):
    total = stats["benign"] + stats["attack"]
    attack_rate = stats["attack"] / total * 100

    print(f"\n{day}")
    print(f"  Total flows: {total:,}")
    print(f"  Benign: {stats['benign']:,}")
    print(f"  Attack: {stats['attack']:,}")
    print(f"  Attack percentage: {attack_rate:.2f}%")

print("\n" + "=" * 70)
print("SOURCE-FILE DISTRIBUTION")
print("=" * 70)

for file, stats in sorted(file_stats.items()):
    total = stats["benign"] + stats["attack"]
    attack_rate = stats["attack"] / total * 100

    print(f"\n{file}")
    print(f"  Total: {total:,}")
    print(f"  Benign: {stats['benign']:,}")
    print(f"  Attack: {stats['attack']:,}")
    print(f"  Attack percentage: {attack_rate:.2f}%")

print("\nAnalysis completed!")
