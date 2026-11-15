
import pandas as pd
import joblib
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

TRAIN_FILE = Path("data/processed/train.csv")
VALIDATION_FILE = Path("data/processed/validation.csv")

MODEL_FILE = Path("models/random_forest_baseline.joblib")

METADATA_COLUMNS = ["Label", "Source_Day", "Source_File"]
TARGET_COLUMN = "Target"

print("=" * 65)
print("TRAINING RANDOM FOREST BASELINE")
print("=" * 65)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("\nLoading training and validation data...")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VALIDATION_FILE)

feature_columns = [
    col for col in train_df.columns
    if col not in METADATA_COLUMNS + [TARGET_COLUMN]
]

X_train = train_df[feature_columns]
y_train = train_df[TARGET_COLUMN]

X_val = val_df[feature_columns]
y_val = val_df[TARGET_COLUMN]

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Network features:", len(feature_columns))

# --------------------------------------------------
# BUILD PIPELINE
# --------------------------------------------------

print("\nBuilding preprocessing and model pipeline...")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        max_features="sqrt"
    ))
])

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\nTraining Random Forest...")
print("This may take some time.")

pipeline.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# VALIDATE
# --------------------------------------------------

print("\nGenerating validation predictions...")

predictions = pipeline.predict(X_val)
probabilities = pipeline.predict_proba(X_val)[:, 1]

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_val,
    predictions,
    target_names=["BENIGN", "ATTACK"],
    digits=4
))

print("CONFUSION MATRIX")
print(confusion_matrix(y_val, predictions))

print("\nROC-AUC:", round(roc_auc_score(y_val, probabilities), 4))
print("PR-AUC:", round(average_precision_score(y_val, probabilities), 4))

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "pipeline": pipeline,
    "feature_columns": feature_columns
}, MODEL_FILE)

print("\nModel saved to:", MODEL_FILE)
print("\nBASELINE TRAINING COMPLETED")
