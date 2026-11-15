
import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path("data/processed/model_ready_dataset.csv")
RESULT_DIR = Path("results")
CHUNK_SIZE = 100_000

print("=" * 70)
print("FRIDAY vs TRAINING-DAY FEATURE DRIFT ANALYSIS")
print("=" * 70)

# Store numeric feature values separately for each group.
training_parts = []
friday_parts = []

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
    training = chunk[
        chunk["Source_Day"].astype(str).str.strip() != "Friday"
    ]
    friday = chunk[
        chunk["Source_Day"].astype(str).str.strip() == "Friday"
    ]

    if not training.empty:
        training_parts.append(training)

    if not friday.empty:
        friday_parts.append(friday)

print("\nCombining data...")
training_df = pd.concat(training_parts, ignore_index=True)
friday_df = pd.concat(friday_parts, ignore_index=True)

# Exclude metadata and target columns.
excluded = ["Label", "Source_Day", "Source_File", "Target"]
features = [
    col for col in training_df.columns
    if col not in excluded
    and pd.api.types.is_numeric_dtype(training_df[col])
]

print("Training records:", f"{len(training_df):,}")
print("Friday records:", f"{len(friday_df):,}")
print("Numeric features:", len(features))

results = []

for feature in features:
    train_values = training_df[feature].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    friday_values = friday_df[feature].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(train_values) == 0 or len(friday_values) == 0:
        continue

    train_median = train_values.median()
    friday_median = friday_values.median()

    train_mean = train_values.mean()
    friday_mean = friday_values.mean()

    train_std = train_values.std()
    friday_std = friday_values.std()

    # Standardized mean difference.
    pooled_std = np.sqrt(
        (train_std ** 2 + friday_std ** 2) / 2
    )

    if pooled_std == 0 or pd.isna(pooled_std):
        standardized_difference = 0.0
    else:
        standardized_difference = (
            abs(friday_mean - train_mean) / pooled_std
        )

    # Relative median change, with safe handling of zero.
    denominator = max(abs(train_median), 1e-9)
    median_change_percent = (
        abs(friday_median - train_median) / denominator
    ) * 100

    results.append({
        "Feature": feature,
        "Training_Mean": train_mean,
        "Friday_Mean": friday_mean,
        "Training_Median": train_median,
        "Friday_Median": friday_median,
        "Training_Std": train_std,
        "Friday_Std": friday_std,
        "Standardized_Mean_Difference": standardized_difference,
        "Median_Change_Percent": median_change_percent
    })

drift_df = pd.DataFrame(results)

drift_df = drift_df.sort_values(
    "Standardized_Mean_Difference",
    ascending=False
)

output_file = RESULT_DIR / "friday_feature_drift.csv"
drift_df.to_csv(output_file, index=False)

print("\nTOP 20 FEATURES WITH THE LARGEST DISTRIBUTION DIFFERENCES")
print("=" * 70)

print(
    drift_df[
        [
            "Feature",
            "Training_Mean",
            "Friday_Mean",
            "Standardized_Mean_Difference",
            "Median_Change_Percent"
        ]
    ].head(20).to_string(index=False)
)

print("\nSaved report:", output_file)
print("\nFEATURE DRIFT ANALYSIS COMPLETED")
