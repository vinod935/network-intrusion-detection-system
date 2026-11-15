
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path("data/processed/model_ready_dataset.csv")
OUTPUT_FILE = Path("data/processed/model_development_sample.csv")

SAMPLE_SIZE = 300_000
CHUNK_SIZE = 200_000
SEED = 42
HELD_OUT_DAY = "Friday"
BUFFER_PER_CLASS = 5_000

# --------------------------------------------------
# PASS 1: COUNT AVAILABLE TRAINING RECORDS
# --------------------------------------------------

print("=" * 65)
print("CREATING MODEL DEVELOPMENT SAMPLE")
print("=" * 65)

print("\nPASS 1: Counting records outside Friday...")

class_counts = {0: 0, 1: 0}

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["Target", "Source_Day"],
    chunksize=CHUNK_SIZE
):
    training_rows = chunk[chunk["Source_Day"] != HELD_OUT_DAY]

    counts = training_rows["Target"].value_counts()

    for target in class_counts:
        class_counts[target] += int(counts.get(target, 0))

total_available = sum(class_counts.values())

if total_available < SAMPLE_SIZE:
    raise ValueError("Not enough records for the requested sample.")

print("\nAvailable training records:")
print("Benign:", f"{class_counts[0]:,}")
print("Attack:", f"{class_counts[1]:,}")
print("Total:", f"{total_available:,}")

# Calculate proportional sample sizes.
quotas = {
    0: round(SAMPLE_SIZE * class_counts[0] / total_available),
    1: 0
}
quotas[1] = SAMPLE_SIZE - quotas[0]

print("\nRequired sample:")
print("Benign:", f"{quotas[0]:,}")
print("Attack:", f"{quotas[1]:,}")

# --------------------------------------------------
# PASS 2: COLLECT A RANDOM CANDIDATE SAMPLE
# --------------------------------------------------

print("\nPASS 2: Sampling records in chunks...")

candidate_samples = []

for chunk_number, chunk in enumerate(
    pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)
):
    training_rows = chunk[
        chunk["Source_Day"] != HELD_OUT_DAY
    ]

    for target in [0, 1]:

        class_rows = training_rows[
            training_rows["Target"] == target
        ]

        if class_rows.empty:
            continue

        fraction = min(
            1.0,
            (quotas[target] + BUFFER_PER_CLASS)
            / class_counts[target]
        )

        selected = class_rows.sample(
            frac=fraction,
            random_state=SEED + chunk_number
        )

        if not selected.empty:
            candidate_samples.append(selected)

    print(f"Processed chunk: {chunk_number + 1}")

# --------------------------------------------------
# PASS 3: SELECT EXACT STRATIFIED SAMPLE
# --------------------------------------------------

print("\nPASS 3: Finalizing sample...")

candidates = pd.concat(
    candidate_samples,
    ignore_index=True
)

final_parts = []

for target in [0, 1]:

    class_candidates = candidates[
        candidates["Target"] == target
    ]

    if len(class_candidates) < quotas[target]:
        raise RuntimeError(
            f"Not enough candidate records for class {target}. "
            "Increase BUFFER_PER_CLASS and run again."
        )

    final_parts.append(
        class_candidates.sample(
            n=quotas[target],
            random_state=SEED
        )
    )

sample = pd.concat(final_parts, ignore_index=True)

# Shuffle records so classes are not grouped.
sample = sample.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

assert len(sample) == SAMPLE_SIZE
assert HELD_OUT_DAY not in sample["Source_Day"].unique()
assert set(sample["Target"].unique()) == {0, 1}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("SAMPLE CREATED SUCCESSFULLY")
print("=" * 65)

print("Total records:", f"{len(sample):,}")
print("Benign records:", f"{(sample['Target'] == 0).sum():,}")
print("Attack records:", f"{(sample['Target'] == 1).sum():,}")
print("Days included:", sorted(sample["Source_Day"].unique()))
print("Friday included:", HELD_OUT_DAY in sample["Source_Day"].unique())
print("Output:", OUTPUT_FILE)
print("File size (MB):", round(OUTPUT_FILE.stat().st_size / (1024 * 1024), 2))
