
import pandas as pd
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

DATA_PATH = "data/processed/model_ready_dataset.csv"

MODEL_A = "models/random_forest_baseline.joblib"
MODEL_B = "models/random_forest_without_destination_port.joblib"

CHUNK_SIZE = 100000


def evaluate_model(model_path, model_name):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    saved = joblib.load(model_path)

    pipeline = saved["pipeline"]
    features = saved["feature_columns"]

    y_actual = []
    y_predicted = []
    y_probabilities = []
    original_labels = []

    reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE)

    for chunk in reader:

        friday = chunk[
            chunk["Source_Day"].astype(str).str.strip() == "Friday"
        ]

        if friday.empty:
            continue

        X = friday[features]
        y = friday["Target"]

        predictions = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)[:, 1]

        y_actual.extend(y.tolist())
        y_predicted.extend(predictions.tolist())
        y_probabilities.extend(probabilities.tolist())
        original_labels.extend(friday["Label"].astype(str).tolist())

    actual = pd.Series(y_actual)
    predicted = pd.Series(y_predicted)
    probabilities = pd.Series(y_probabilities)
    labels = pd.Series(original_labels)

    print("\nFriday records:", len(actual))
    print("Features used:", len(features))

    print("\nCLASSIFICATION REPORT")
    print("-" * 70)

    print(classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        digits=4,
        zero_division=0
    ))

    print("Accuracy:", accuracy_score(actual, predicted))
    print("Macro F1:", f1_score(actual, predicted, average="macro"))
    print("PR-AUC:", average_precision_score(actual, probabilities))
    print("ROC-AUC:", roc_auc_score(actual, probabilities))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))

    # Attack recall by original attack type
    print("\nATTACK DETECTION BY ORIGINAL LABEL")
    print("-" * 70)

    results = pd.DataFrame({
        "Label": labels,
        "Actual": actual,
        "Predicted": predicted
    })

    for label in sorted(results["Label"].unique()):

        group = results[
            (results["Label"] == label) &
            (results["Actual"] == 1)
        ]

        if group.empty:
            continue

        detected = (group["Predicted"] == 1).sum()
        total = len(group)
        recall = detected / total

        print(
            f"{label}: "
            f"Total={total}, "
            f"Detected={detected}, "
            f"Recall={recall:.4%}"
        )

    # False positive rate
    benign = actual == 0
    false_positives = ((predicted == 1) & benign).sum()
    total_benign = benign.sum()

    fpr = false_positives / total_benign

    print("\nFalse Positive Rate:", round(fpr, 6))


evaluate_model(MODEL_A, "MODEL A: ORIGINAL 69 FEATURES")

evaluate_model(MODEL_B, "MODEL B: WITHOUT DESTINATION PORT")

print("\n" + "=" * 70)
print("FRIDAY MODEL COMPARISON COMPLETED")
print("=" * 70)
