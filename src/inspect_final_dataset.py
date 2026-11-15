
import pandas as pd
from pathlib import Path

FILE = Path("data/processed/binary_dataset_final.csv")

print("=" * 70)
print("FINAL DATASET INSPECTION")
print("=" * 70)

df = pd.read_csv(FILE, nrows=10)

print("\nTotal columns:", len(df.columns))
print("\nColumn names:")

for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

print("\nFirst 5 records:")
print(df.head().to_string())

print("\nTarget distribution:")

target_counts = pd.read_csv(
    FILE,
    usecols=["Target"]
)["Target"].value_counts().sort_index()

print(target_counts)

print("\nDataset metadata:")
print("File size (MB):", round(FILE.stat().st_size / (1024 * 1024), 2))
print("Duplicate feature present:", "Fwd Header Length.1" in df.columns)

print("\nInspection completed.")
