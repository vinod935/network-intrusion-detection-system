
import pandas as pd
import numpy as np

DATA_PATH = "data/processed/model_ready_dataset.csv"

FEATURES = [
    "Destination Port",
    "Fwd Header Length",
    "Init_Win_bytes_forward",
    "Fwd IAT Min",
    "Flow Duration",
    "Total Fwd Packets",
    "Average Packet Size"
]

CHUNK_SIZE = 100000

# Store only required feature values
groups = {
    "Training_Benign": [],
    "Training_Attack": [],
    "Friday_Benign": [],
    "Friday_Attack": []
}

print("=" * 70)
print("TRAINING VS FRIDAY FEATURE DISTRIBUTION")
print("=" * 70)

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):

    day = chunk["Source_Day"].astype(str).str.strip()
    target = chunk["Target"]

    masks = {
        "Training_Benign": (day != "Friday") & (target == 0),
        "Training_Attack": (day != "Friday") & (target == 1),
        "Friday_Benign": (day == "Friday") & (target == 0),
        "Friday_Attack": (day == "Friday") & (target == 1)
    }

    for group_name, mask in masks.items():

        selected = chunk.loc[mask, FEATURES]

        if not selected.empty:
            groups[group_name].append(selected)

print("\nAll records collected. Calculating statistics...")

for feature in FEATURES:

    print("\n" + "=" * 70)
    print("FEATURE:", feature)
    print("-" * 70)

    for group_name, frames in groups.items():

        if not frames:
            continue

        values = pd.concat(
            [frame[feature] for frame in frames],
            ignore_index=True
        )

        values = pd.to_numeric(
            values,
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        print(
            f"{group_name}: "
            f"count={len(values)}, "
            f"median={values.median():.4f}, "
            f"mean={values.mean():.4f}, "
            f"25%={values.quantile(0.25):.4f}, "
            f"75%={values.quantile(0.75):.4f}"
        )

print("\n" + "=" * 70)
print("DISTRIBUTION COMPARISON COMPLETED")
print("=" * 70)
