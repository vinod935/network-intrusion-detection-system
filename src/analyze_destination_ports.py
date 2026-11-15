
import pandas as pd

TRAIN_PATH = "data/processed/train.csv"
FULL_PATH = "data/processed/model_ready_dataset.csv"

print("=" * 70)
print("DESTINATION PORT ANALYSIS")
print("=" * 70)

# Training data
train = pd.read_csv(TRAIN_PATH, usecols=["Destination Port", "Target"])

print("\n1. TOP DESTINATION PORTS IN TRAINING DATA")
print("-" * 70)

for target, name in [(0, "BENIGN"), (1, "ATTACK")]:
    subset = train[train["Target"] == target]

    print(f"\n{name} traffic:")

    print(
        subset["Destination Port"]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

# Friday data
print("\n\n2. TOP DESTINATION PORTS IN FRIDAY DATA")
print("-" * 70)

friday_chunks = []

for chunk in pd.read_csv(
    FULL_PATH,
    usecols=["Destination Port", "Target", "Source_Day"],
    chunksize=100000
):
    friday = chunk[chunk["Source_Day"] == "Friday"]

    if not friday.empty:
        friday_chunks.append(
            friday[["Destination Port", "Target"]]
        )

friday_data = pd.concat(friday_chunks, ignore_index=True)

for target, name in [(0, "BENIGN"), (1, "ATTACK")]:
    subset = friday_data[friday_data["Target"] == target]

    print(f"\nFriday {name} traffic:")

    print(
        subset["Destination Port"]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

print("\n\nAnalysis completed.")
