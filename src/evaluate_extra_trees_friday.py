
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

print("=" * 65)
print("EXTRA TREES - FRIDAY DIAGNOSTIC EVALUATION")
print("=" * 65)

# Load saved model
saved = joblib.load("models/extra_trees_baseline.joblib")

pipeline = saved["pipeline"]
feature_columns = saved["feature_columns"]

# Load Friday records
df = pd.read_csv(
    "data/processed/model_ready_dataset.csv"
)

friday = df[df["Source_Day"] == "Friday"].copy()

X = friday[feature_columns]
y = friday["Target"]

print("\nFriday records:", len(friday))
print("Features:", len(feature_columns))

# Predict
print("\nGenerating predictions...")

predictions = pipeline.predict(X)
probabilities = pipeline.predict_proba(X)[:, 1]

# Overall metrics
print("\nCLASSIFICATION REPORT")
print("-" * 65)

print(classification_report(
    y,
    predictions,
    target_names=["BENIGN", "ATTACK"],
    digits=4
))

print("Accuracy:", accuracy_score(y, predictions))
print("Macro F1:", f1_score(y, predictions, average="macro"))
print("ROC-AUC:", roc_auc_score(y, probabilities))
print("PR-AUC:", average_precision_score(y, probabilities))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y, predictions))

# Per-attack-family recall
print("\nPER-ATTACK-FAMILY RECALL")
print("-" * 65)

results = friday[["Label"]].copy()
results["Actual"] = y.values
results["Predicted"] = predictions

for label in sorted(results["Label"].unique()):
    subset = results[results["Label"] == label]

    actual_attacks = subset["Actual"].sum()
    detected_attacks = (
        (subset["Actual"] == 1) &
        (subset["Predicted"] == 1)
    ).sum()

    if actual_attacks > 0:
        recall = detected_attacks / actual_attacks
        print(
            f"{label}: "
            f"{detected_attacks}/{actual_attacks} "
            f"({recall:.4%})"
        )

print("\nFRIDAY EVALUATION COMPLETED")
