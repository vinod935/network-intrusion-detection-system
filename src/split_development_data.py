
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

INPUT_FILE = Path("data/processed/model_development_sample.csv")
TRAIN_FILE = Path("data/processed/train.csv")
VALIDATION_FILE = Path("data/processed/validation.csv")

SEED = 42

METADATA_COLUMNS = ["Label", "Source_Day", "Source_File"]
TARGET_COLUMN = "Target"

print("=" * 65)
print("CREATING LEAKAGE-FREE TRAINING AND VALIDATION SETS")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

# Keep all records, including metadata, in the split.
train_df, validation_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df[TARGET_COLUMN]
)

# Identify only the 69 network features as model inputs.
feature_columns = [
    col for col in df.columns
    if col not in METADATA_COLUMNS + [TARGET_COLUMN]
]

# Verify that metadata is not included in model features.
assert len(feature_columns) == 69
assert not set(feature_columns).intersection(METADATA_COLUMNS)

# Verify that the target is not included in model features.
assert TARGET_COLUMN not in feature_columns

TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

train_df.to_csv(TRAIN_FILE, index=False)
validation_df.to_csv(VALIDATION_FILE, index=False)

print("\nSPLIT COMPLETED")
print("-" * 65)

print("Training records:", f"{len(train_df):,}")
print("Validation records:", f"{len(validation_df):,}")
print("Network features:", len(feature_columns))
print("Metadata columns:", METADATA_COLUMNS)

print("\nTraining class distribution:")
print(train_df[TARGET_COLUMN].value_counts().sort_index())

print("\nValidation class distribution:")
print(validation_df[TARGET_COLUMN].value_counts().sort_index())

print("\nTraining days:", sorted(train_df["Source_Day"].unique()))
print("Validation days:", sorted(validation_df["Source_Day"].unique()))

print("\nFriday in training:",
      "Friday" in train_df["Source_Day"].unique())

print("Friday in validation:",
      "Friday" in validation_df["Source_Day"].unique())

print("\nModel input columns:")
print(feature_columns)

print("\nFiles saved:")
print(TRAIN_FILE)
print(VALIDATION_FILE)

print("\nMetadata is preserved but excluded from model inputs.")
print("Leakage checks passed.")
