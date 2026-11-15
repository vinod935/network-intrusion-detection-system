
import pandas as pd
import numpy as np
import joblib
import time

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

print("=" * 65)
print("HISTOGRAM GRADIENT BOOSTING CLASSIFIER")
print("=" * 65)

# Load datasets
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/validation.csv")

# Identify input features
excluded_columns = [
    "Target",
    "Label",
    "Source_Day",
    "Source_File"
]

feature_columns = [
    col for col in train_df.columns
    if col not in excluded_columns
]

X_train = train_df[feature_columns]
y_train = train_df["Target"]

X_val = val_df[feature_columns]
y_val = val_df["Target"]

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Features:", len(feature_columns))

# Create model pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("model", HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42
    ))
])

# Train
print("\nTraining HistGradientBoosting...")
start_time = time.time()

pipeline.fit(X_train, y_train)

elapsed = time.time() - start_time

print(f"Training completed in {elapsed:.2f} seconds.")

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

# Save model and feature order
joblib.dump(
    {
        "pipeline": pipeline,
        "feature_columns": feature_columns
    },
    "models/hist_gradient_boosting.joblib"
)

print("\nModel saved: models/hist_gradient_boosting.joblib")
print("\nHISTGRADIENTBOOSTING TRAINING COMPLETED")
