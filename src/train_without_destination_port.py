
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

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

MODEL_PATH = "models/random_forest_without_destination_port.joblib"

print("=" * 65)
print("RANDOM FOREST WITHOUT DESTINATION PORT")
print("=" * 65)

# Load datasets
train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)

# Exclude target and metadata
excluded = ["Target", "Label", "Source_Day", "Source_File"]

feature_columns = [
    col for col in train.columns
    if col not in excluded
]

# Remove Destination Port
feature_columns.remove("Destination Port")

X_train = train[feature_columns]
y_train = train["Target"]

X_val = validation[feature_columns]
y_val = validation["Target"]

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Features:", len(feature_columns))
print("Destination Port removed:", 
      "Destination Port" not in feature_columns)

# Build model
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

print("\nTraining model...")

pipeline.fit(X_train, y_train)

print("Training completed.")

# Predict
predictions = pipeline.predict(X_val)
probabilities = pipeline.predict_proba(X_val)[:, 1]

# Evaluate
print("\nCLASSIFICATION REPORT")
print("-" * 65)

print(classification_report(
    y_val,
    predictions,
    target_names=["BENIGN", "ATTACK"],
    digits=4
))

print("Accuracy:", accuracy_score(y_val, predictions))
print("Macro F1:", f1_score(y_val, predictions, average="macro"))
print("PR-AUC:", average_precision_score(y_val, probabilities))
print("ROC-AUC:", roc_auc_score(y_val, probabilities))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_val, predictions))

# Save model and feature names
joblib.dump({
    "pipeline": pipeline,
    "feature_columns": feature_columns
}, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("\nExperiment completed.")
