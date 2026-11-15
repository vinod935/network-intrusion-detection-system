import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path("data/processed/family_aware_sample.csv")

TRAIN_FILE = Path("data/processed/train_family_aware.csv")
VALIDATION_FILE = Path("data/processed/validation_family_aware.csv")

TEST_SIZE = 0.20
SEED = 42

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 65)
print("FAMILY-AWARE GROUP-BASED SPLIT")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

metadata = ["Label", "Source_Day", "Source_File", "Target"]
features = [column for column in df.columns if column not in metadata]

print("\nDataset shape:", df.shape)
print("Feature count:", len(features))

# --------------------------------------------------
# CREATE FEATURE GROUPS
# --------------------------------------------------

print("\nCreating feature groups...")

groups = pd.util.hash_pandas_object(
    df[features],
    index=False
)

print("Unique feature groups:", groups.nunique())

# --------------------------------------------------
# GROUP-BASED SPLIT
# --------------------------------------------------

print("\nSplitting into training and validation sets...")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=SEED
)

train_indices, validation_indices = next(
    splitter.split(df, groups=groups)
)

train_df = df.iloc[train_indices].copy()
validation_df = df.iloc[validation_indices].copy()

# --------------------------------------------------
# VALIDATION CHECKS
# --------------------------------------------------

train_groups = set(groups.iloc[train_indices])
validation_groups = set(groups.iloc[validation_indices])

overlap = train_groups.intersection(validation_groups)

assert len(overlap) == 0
assert len(train_df) + len(validation_df) == len(df)

assert set(train_df["Target"].unique()) == {0, 1}
assert set(validation_df["Target"].unique()) == {0, 1}

assert "Friday" not in train_df["Source_Day"].unique()
assert "Friday" not in validation_df["Source_Day"].unique()

# --------------------------------------------------
# SAVE FILES
# --------------------------------------------------

TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

train_df.to_csv(TRAIN_FILE, index=False)
validation_df.to_csv(VALIDATION_FILE, index=False)

# --------------------------------------------------
# REPORT
# --------------------------------------------------

print("\n" + "=" * 65)
print("SPLIT SUMMARY")
print("=" * 65)

print("\nTraining records:", f"{len(train_df):,}")
print("Validation records:", f"{len(validation_df):,}")

print("\nTraining target distribution:")
print(train_df["Target"].value_counts().to_string())

print("\nValidation target distribution:")
print(validation_df["Target"].value_counts().to_string())

print("\nTraining attack families:")
print(train_df["Label"].value_counts().to_string())

print("\nValidation attack families:")
print(validation_df["Label"].value_counts().to_string())

print("\nOverlapping feature groups:", len(overlap))
print("Friday in training:", "Friday" in train_df["Source_Day"].unique())
print("Friday in validation:", "Friday" in validation_df["Source_Day"].unique())

print("\nTraining file:", TRAIN_FILE)
print("Validation file:", VALIDATION_FILE)

print("\nSplit completed successfully.")

