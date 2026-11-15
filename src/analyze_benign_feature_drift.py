
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

DATA_FILE = Path("data/processed/model_ready_dataset.csv")
RESULT_DIR = Path("results")
CHUNK_SIZE = 100_000
SAMPLE_SIZE = 100_000

training_parts = []
friday_parts = []

print("=" * 70)
print("BENIGN-ONLY FEATURE DRIFT ANALYSIS")
print("=" * 70)

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
    benign = chunk[chunk["Target"] == 0]

    training = benign[
        benign["Source_Day"].astype(str).str.strip() != "Friday"
    ]
    friday = benign[
        benign["Source_Day"].astype(str).str.strip() == "Friday"
    ]

    if not training.empty:
        training_parts.append(training)

    if not friday.empty:
        friday_parts.append(friday)

training_df = pd.concat(training_parts, ignore_index=True)
friday_df = pd.concat(friday_parts, ignore_index=True)

excluded = ["Label", "Source_Day", "Source_File", "Target"]
features = [
    col for col in training_df.columns
    if col not in excluded
    and pd.api.types.is_numeric_dtype(training_df[col])
]

print("Training BENIGN records:", f"{len(training_df):,}")
print("Friday BENIGN records:", f"{len(friday_df):,}")
print("Features:", len(features))

results = []

for feature in features:
    train = training_df[feature].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    friday = friday_df[feature].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(train) == 0 or len(friday) == 0:
        continue

    # Sample for the two-sample KS test.
    train_sample = train.sample(
        min(SAMPLE_SIZE, len(train)), random_state=42
    )
    friday_sample = friday.sample(
        min(SAMPLE_SIZE, len(friday)), random_state=42
    )

    ks_stat, p_value = ks_2samp(
        train_sample, friday_sample
    )

    train_q25 = train.quantile(0.25)
    train_median = train.median()
    train_q75 = train.quantile(0.75)

    friday_q25 = friday.quantile(0.25)
    friday_median = friday.median()
    friday_q75 = friday.quantile(0.75)

    results.append({
        "Feature": feature,
        "Training_Median": train_median,
        "Friday_Median": friday_median,
        "Training_Q25": train_q25,
        "Training_Q75": train_q75,
        "Friday_Q25": friday_q25,
        "Friday_Q75": friday_q75,
        "KS_Statistic": ks_stat,
        "KS_P_Value": p_value
    })

result_df = pd.DataFrame(results).sort_values(
    "KS_Statistic", ascending=False
)

output = RESULT_DIR / "friday_benign_feature_drift.csv"
result_df.to_csv(output, index=False)

print("\nTOP 20 BENIGN FEATURES WITH DISTRIBUTION DIFFERENCES")
print("=" * 70)

print(result_df.head(20).to_string(index=False))

print("\nSaved:", output)
print("BENIGN-ONLY ANALYSIS COMPLETED")
