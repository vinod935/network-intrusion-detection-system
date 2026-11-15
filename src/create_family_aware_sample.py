import pandas as pd
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path("data/processed/model_ready_dataset.csv")
OUTPUT_FILE = Path("data/processed/family_aware_sample.csv")

SAMPLE_SIZE = 300_000
CHUNK_SIZE = 200_000
SEED = 42
HELD_OUT_DAY = "Friday"

# Keep all available records for rare attack families.
# Sample common families to reach the target size.
BENIGN_QUOTA = 250_000
DOS_HULK_QUOTA = 12_350

# --------------------------------------------------
# PASS 1: COUNT AVAILABLE RECORDS BY FAMILY
# --------------------------------------------------

print("=" * 65)
print("CREATING FAMILY-AWARE DEVELOPMENT SAMPLE")
print("=" * 65)

print("\nPASS 1: Counting records by attack family...")

family_counts = {}

for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["Label", "Source_Day"],
    chunksize=CHUNK_SIZE
):
    training_rows = chunk[
        chunk["Source_Day"].astype(str).str.strip() != HELD_OUT_DAY
    ]

    counts = training_rows["Label"].value_counts()

    for label, count in counts.items():
        family_counts[label] = (
            family_counts.get(label, 0) + int(count)
        )

print("\nAvailable records:")
for label, count in sorted(
    family_counts.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{label}: {count:,}")

# --------------------------------------------------
# PASS 2: DEFINE FAMILY QUOTAS
# --------------------------------------------------

quotas = {}

for label, available in family_counts.items():

    if label == "BENIGN":
        quotas[label] = BENIGN_QUOTA

    elif label == "DoS Hulk":
        quotas[label] = DOS_HULK_QUOTA

    else:
        # Retain every available record from other families.
        quotas[label] = available

required_total = sum(quotas.values())

if required_total != SAMPLE_SIZE:
    raise ValueError(
        f"Quota total is {required_total:,}, "
        f"not {SAMPLE_SIZE:,}. Review the quotas."
    )

for label, quota in quotas.items():
    if quota > family_counts[label]:
        raise ValueError(
            f"Quota for {label} exceeds available records."
        )

print("\nRequired sample by family:")
for label, quota in sorted(
    quotas.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{label}: {quota:,}")

# --------------------------------------------------
# PASS 3: COLLECT CANDIDATE RECORDS
# --------------------------------------------------

print("\nPASS 3: Collecting candidate records...")

candidate_samples = []

for chunk_number, chunk in enumerate(
    pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)
):
    training_rows = chunk[
        chunk["Source_Day"].astype(str).str.strip() != HELD_OUT_DAY
    ]

    for label, quota in quotas.items():

        if quota == family_counts[label]:
            # All records from this family are required.
            selected = training_rows[
                training_rows["Label"] == label
            ]

        else:
            family_rows = training_rows[
                training_rows["Label"] == label
            ]

            if family_rows.empty:
                continue

            fraction = min(
                1.0,
                (quota + 5_000) / family_counts[label]
            )

            selected = family_rows.sample(
                frac=fraction,
                random_state=SEED + chunk_number
            )

        if not selected.empty:
            candidate_samples.append(selected)

    print(f"Processed chunk: {chunk_number + 1}")

# --------------------------------------------------
# PASS 4: SELECT EXACT QUOTAS
# --------------------------------------------------

print("\nPASS 4: Finalizing family-aware sample...")

candidates = pd.concat(
    candidate_samples,
    ignore_index=True
)

final_parts = []

for label, quota in quotas.items():

    family_candidates = candidates[
        candidates["Label"] == label
    ]

    if len(family_candidates) < quota:
        raise RuntimeError(
            f"Not enough candidates for {label}: "
            f"{len(family_candidates):,} available, "
            f"{quota:,} required."
        )

    final_parts.append(
        family_candidates.sample(
            n=quota,
            random_state=SEED
        )
    )

sample = pd.concat(final_parts, ignore_index=True)

sample = sample.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

assert len(sample) == SAMPLE_SIZE
assert HELD_OUT_DAY not in sample["Source_Day"].astype(str).str.strip().unique()
assert sample["Label"].value_counts().to_dict() == quotas

# Binary target must remain consistent with labels.
assert (
    (sample["Label"] == "BENIGN") == (sample["Target"] == 0)
).all()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
sample.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 65)
print("FAMILY-AWARE SAMPLE CREATED")
print("=" * 65)

print("Total records:", f"{len(sample):,}")
print("\nFinal family distribution:")
print(sample["Label"].value_counts().to_string())

print("\nDays included:", sorted(sample["Source_Day"].unique()))
print("Friday included:", HELD_OUT_DAY in sample["Source_Day"].astype(str).str.strip().unique())
print("Output:", OUTPUT_FILE)
print("File size (MB):", round(OUTPUT_FILE.stat().st_size / (1024 * 1024), 2))
