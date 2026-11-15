
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

INPUT_FILE = Path("data/processed/model_ready_dataset.csv")
MODEL_FILE = Path("models/random_forest_baseline.joblib")

CHUNK_SIZE = 100_000
HELD_OUT_DAY = "Friday"

print("=" * 65)
print("UNSEEN-DAY EVALUATION")
print("=" * 65)

# Load the previously trained model.
saved = joblib.load(MODEL_FILE)

pipeline = saved["pipeline"]
feature_columns = saved["feature_columns"]

print("\nModel loaded.")
print("Features:", len(feature_columns))
print("Testing day:", HELD_OUT_DAY)

# Store predictions from Friday only.
all_actual = []
all_predictions = []
all_probabilities = []
all_sources = []

total_friday_rows = 0

print("\nScanning dataset in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)
):

    friday = chunk[
        chunk["Source_Day"] == HELD_OUT_DAY
    ]

    if friday.empty:
        continue

    X = friday[feature_columns]
    y = friday["Target"]

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    all_actual.extend(y.tolist())
    all_predictions.extend(predictions.tolist())
    all_probabilities.extend(probabilities.tolist())
    all_sources.extend(friday["Source_File"].tolist())

    total_friday_rows += len(friday)

    print(
        f"Friday records evaluated: {total_friday_rows:,}"
    )

# Convert results to Series.
actual = pd.Series(all_actual)
predicted = pd.Series(all_predictions)
probabilities = pd.Series(all_probabilities)
sources = pd.Series(all_sources)

print("\n" + "=" * 65)
print("FRIDAY EVALUATION RESULTS")
print("=" * 65)

print("Total Friday records:", f"{total_friday_rows:,}")

print("\nActual class distribution:")
print(actual.value_counts().sort_index())

print("\nClassification report:")
print(classification_report(
    actual,
    predicted,
    labels=[0, 1],
    target_names=["BENIGN", "ATTACK"],
    digits=4,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(actual, predicted, labels=[0, 1]))

print(
    "\nROC-AUC:",
    round(roc_auc_score(actual, probabilities), 4)
)

print(
    "PR-AUC:",
    round(average_precision_score(actual, probabilities), 4)
)

# --------------------------------------------------
# PER-FILE ERROR ANALYSIS
# --------------------------------------------------

print("\nPER-FILE EVALUATION")

results = pd.DataFrame({
    "Source_File": sources,
    "Actual": actual,
    "Predicted": predicted
})

for filename, group in results.groupby("Source_File"):

    correct = (group["Actual"] == group["Predicted"]).sum()
    accuracy = correct / len(group)

    attack_rows = group[group["Actual"] == 1]

    if len(attack_rows) > 0:
        attack_recall = (
            attack_rows["Predicted"] == 1
        ).mean()
    else:
        attack_recall = None

    benign_rows = group[group["Actual"] == 0]

    if len(benign_rows) > 0:
        false_positive_rate = (
            benign_rows["Predicted"] == 1
        ).mean()
    else:
        false_positive_rate = None

    print("\nFile:", filename)
    print("Records:", f"{len(group):,}")
    print("Accuracy:", round(accuracy, 4))
    print(
        "Attack recall:",
        round(attack_recall, 4)
        if attack_recall is not None else "N/A"
    )
    print(
        "False positive rate:",
        round(false_positive_rate, 4)
        if false_positive_rate is not None else "N/A"
    )

print("\nUNSEEN-DAY EVALUATION COMPLETED")
