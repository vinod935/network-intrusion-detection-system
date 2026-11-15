import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_FILE = Path("data/processed/model_ready_dataset.csv")
MODEL_FILE = Path("models/hist_gradient_boosting_family_aware.joblib")
THRESHOLD_FILE = Path("results/family_aware_hgb_threshold.json")

RESULT_DIR = Path("results")
CHUNK_SIZE = 100_000

# --------------------------------------------------
# LOAD MODEL AND SAVED THRESHOLD
# --------------------------------------------------

print("=" * 70)
print("FRIDAY THRESHOLD EVALUATION")
print("=" * 70)

saved = joblib.load(MODEL_FILE)

pipeline = saved["pipeline"]
features = saved["feature_columns"]

with open(THRESHOLD_FILE, "r") as file:
    threshold_data = json.load(file)

selected_threshold = float(threshold_data["threshold"])

print("Model:", MODEL_FILE)
print("Default threshold: 0.50")
print("Validation-selected threshold:", selected_threshold)

# --------------------------------------------------
# EVALUATE FRIDAY DATA
# --------------------------------------------------

actual_all = []
probability_all = []
labels_all = []

total_records = 0

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):

    friday = chunk[
        chunk["Source_Day"].astype(str).str.strip() == "Friday"
    ]

    if friday.empty:
        continue

    X = friday[features]

    probabilities = pipeline.predict_proba(X)[:, 1]

    actual_all.extend(friday["Target"].tolist())
    probability_all.extend(probabilities.tolist())
    labels_all.extend(friday["Label"].astype(str).tolist())

    total_records += len(friday)

    print(f"Records evaluated: {total_records:,}", end="\r")

actual = np.array(actual_all)
probabilities = np.array(probability_all)
labels = np.array(labels_all)

if total_records == 0:
    raise ValueError("No Friday records found.")

print("\n\nTotal Friday records:", f"{total_records:,}")

# --------------------------------------------------
# COMPARE TWO THRESHOLDS
# --------------------------------------------------

summary = []
family_results = []

thresholds = {
    "Default 0.50": 0.50,
    "Validation-selected": selected_threshold
}

for threshold_name, threshold in thresholds.items():

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(actual, predictions, zero_division=0)
    recall = recall_score(actual, predictions, zero_division=0)
    f1 = f1_score(actual, predictions, zero_division=0)

    fpr = fp / (fp + tn)

    print("\n" + "=" * 65)
    print(threshold_name)
    print("=" * 65)

    print("Threshold:", round(threshold, 6))
    print("Accuracy:", round(accuracy, 6))
    print("Attack precision:", round(precision, 6))
    print("Attack recall:", round(recall, 6))
    print("Attack F1:", round(f1, 6))
    print("False positive rate:", round(fpr, 6))
    print("True negatives:", tn)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True positives:", tp)

    summary.append({
        "Threshold_Type": threshold_name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Attack_Precision": precision,
        "Attack_Recall": recall,
        "Attack_F1": f1,
        "False_Positive_Rate": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "ROC_AUC": roc_auc_score(actual, probabilities),
        "PR_AUC": average_precision_score(actual, probabilities)
    })

    print("\nATTACK-FAMILY RECALL")

    for family in sorted(set(labels[actual == 1])):

        mask = (labels == family) & (actual == 1)

        total = int(mask.sum())
        detected = int((predictions[mask] == 1).sum())

        family_recall = detected / total

        print(
            f"{family}: {detected}/{total}, "
            f"Recall={family_recall:.4%}"
        )

        family_results.append({
            "Threshold_Type": threshold_name,
            "Threshold": threshold,
            "Attack_Family": family,
            "Total_Attacks": total,
            "Detected": detected,
            "Missed": total - detected,
            "Recall": family_recall
        })

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

pd.DataFrame(summary).to_csv(
    RESULT_DIR / "family_aware_friday_threshold_comparison.csv",
    index=False
)

pd.DataFrame(family_results).to_csv(
    RESULT_DIR / "family_aware_friday_threshold_family_recall.csv",
    index=False
)

print("\n" + "=" * 70)
print("FRIDAY THRESHOLD EVALUATION COMPLETED")
print("=" * 70)

print("Saved:")
print("results/family_aware_friday_threshold_comparison.csv")
print("results/family_aware_friday_threshold_family_recall.csv")

