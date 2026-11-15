
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"

MODEL_PATH = "models/extra_trees_baseline.joblib"

print("=" * 65)
print("EXTRA TREES CLASSIFIER")
print("=" * 65)

# Load datasets
train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)

# Exclude metadata and target
excluded = [
    "Target",
    "Label",
    "Source_Day",
    "Source_File"
]

feature_columns = [
    col for col in train.columns
    if col not in excluded
]

X_train = train[feature_columns]
y_train = train["Target"]

X_val = validation[feature_columns]
y_val = validation["Target"]

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Features:", len(feature_columns))

# Build Extra Trees pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", ExtraTreesClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        max_features="sqrt"
    ))
])

print("\nTraining Extra Trees...")

pipeline.fit(X_train, y_train)

print("Training completed.")

# Predictions
predictions = pipeline.predict(X_val)
probabilities = pipeline.predict_proba(X_val)[:, 1]

# Evaluation
print("\nCLASSIFICATION REPORT")
print("-" * 65)

print(classification_report(
    y_val,
    predictions,
    target_names=["BENIGN", "ATTACK"],
    digits=4
))

print("\nAccuracy:",
      accuracy_score(y_val, predictions))

print("Macro F1:",
      f1_score(y_val, predictions, average="macro"))

print("PR-AUC:",
      average_precision_score(y_val, probabilities))

print("ROC-AUC:",
      roc_auc_score(y_val, probabilities))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_val, predictions))

# Save model
joblib.dump({
    "pipeline": pipeline,
    "feature_columns": feature_columns
}, MODEL_PATH)

print("\nModel saved:", MODEL_PATH)
print("\nEXTRA TREES TRAINING COMPLETED")
