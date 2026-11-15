import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

VAL_FILE = Path("data/processed/validation_family_aware.csv")

MODEL_FILE = Path("models/hist_gradient_boosting_family_aware.joblib")

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

TARGET_RECALL = 0.95

# --------------------------------------------------
# LOAD MODEL AND VALIDATION DATA
# --------------------------------------------------

print("=" * 70)
print("FAMILY-AWARE HIST GRADIENT BOOSTING THRESHOLD ANALYSIS")
print("=" * 70)

saved = joblib.load(MODEL_FILE)

pipeline = saved["pipeline"]
features = saved["feature_columns"]

val_df = pd.read_csv(VAL_FILE)

X_val = val_df[features]
y_val = val_df["Target"]

print("\nModel:", MODEL_FILE)
print("Validation records:", len(val_df))
print("Features:", len(features))

if "Friday" in val_df["Source_Day"].astype(str).unique():
    raise ValueError("Friday must not appear in validation data.")

# --------------------------------------------------
# CALCULATE ATTACK PROBABILITIES
# --------------------------------------------------

print("\nCalculating validation probabilities...")

val_prob = pipeline.predict_proba(X_val)[:, 1]

# --------------------------------------------------
# SELECT THRESHOLD USING VALIDATION ONLY
# --------------------------------------------------

precision_curve, recall_curve, thresholds = precision_recall_curve(
    y_val,
    val_prob
)

eligible = np.where(recall_curve[:-1] >= TARGET_RECALL)[0]

if len(eligible) == 0:
    raise ValueError(
        "No threshold achieves the target validation recall."
    )

# Highest threshold that still achieves at least 95% recall.
index = eligible[-1]
selected_threshold = float(thresholds[index])

print("\n" + "=" * 70)
print("SELECTED THRESHOLD")
print("=" * 70)

print("Target attack recall:", TARGET_RECALL)
print("Selected threshold:", round(selected_threshold, 6))
print("Validation precision:", round(precision_curve[index], 6))
print("Validation recall:", round(recall_curve[index], 6))

# --------------------------------------------------
# COMPARE DEFAULT AND SELECTED THRESHOLDS
# --------------------------------------------------

results = []

for threshold_name, current_threshold in [
    ("Default 0.50", 0.50),
    ("Validation-selected", selected_threshold)
]:

    predictions = (val_prob >= current_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    print("\n" + "-" * 60)
    print(threshold_name)
    print("Threshold:", round(current_threshold, 6))
    print("Accuracy:", round(accuracy, 6))
    print("Attack precision:", round(precision, 6))
    print("Attack recall:", round(recall, 6))
    print("Attack F1:", round(f1, 6))
    print("False positive rate:", round(fpr, 6))
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)

    results.append({
        "Threshold_Type": threshold_name,
        "Threshold": current_threshold,
        "Accuracy": accuracy,
        "Attack_Precision": precision,
        "Attack_Recall": recall,
        "Attack_F1": f1,
        "False_Positive_Rate": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

# --------------------------------------------------
# SAVE SELECTED THRESHOLD AND RESULTS
# --------------------------------------------------

threshold_file = RESULT_DIR / "family_aware_hgb_threshold.json"

with open(threshold_file, "w") as file:
    json.dump({
        "model": "Hist Gradient Boosting Family-Aware",
        "threshold": selected_threshold,
        "target_validation_recall": TARGET_RECALL
    }, file, indent=4)

results_file = RESULT_DIR / "family_aware_hgb_threshold_comparison.csv"

pd.DataFrame(results).to_csv(results_file, index=False)

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 70)

print("Selected threshold saved:", threshold_file)
print("Comparison saved:", results_file)

