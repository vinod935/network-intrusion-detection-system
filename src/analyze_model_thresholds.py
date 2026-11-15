
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

print("=" * 75)
print("MODEL THRESHOLD ANALYSIS")
print("=" * 75)

# Load validation and Friday datasets
val_df = pd.read_csv("data/processed/validation.csv")

df = pd.read_csv("data/processed/model_ready_dataset.csv")
friday_df = df[df["Source_Day"] == "Friday"].copy()

models = {
    "Random Forest": "models/random_forest_baseline.joblib",
    "Extra Trees": "models/extra_trees_baseline.joblib",
    "HistGradientBoosting": "models/hist_gradient_boosting.joblib"
}

TARGET_RECALL = 0.95

for model_name, model_path in models.items():

    print("\n" + "=" * 75)
    print(model_name.upper())
    print("=" * 75)

    saved = joblib.load(model_path)

    pipeline = saved["pipeline"]
    features = saved["feature_columns"]

    X_val = val_df[features]
    y_val = val_df["Target"]

    X_friday = friday_df[features]
    y_friday = friday_df["Target"]

    # Obtain attack probabilities
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    friday_prob = pipeline.predict_proba(X_friday)[:, 1]

    # Select threshold using validation data only
    precision, recall, thresholds = precision_recall_curve(
        y_val,
        val_prob
    )

    eligible = np.where(recall[:-1] >= TARGET_RECALL)[0]

    if len(eligible) == 0:
        print("Could not find threshold for target recall.")
        continue

    # Highest threshold that still achieves target recall
    index = eligible[-1]
    threshold = thresholds[index]

    print(f"\nSelected threshold: {threshold:.6f}")
    print(f"Validation recall at selection: {recall[index]:.4f}")
    print(f"Validation precision at selection: {precision[index]:.4f}")

    # Compare default and selected thresholds
    for dataset_name, y_true, probabilities in [
        ("VALIDATION", y_val, val_prob),
        ("FRIDAY DIAGNOSTIC", y_friday, friday_prob)
    ]:

        print(f"\n{dataset_name}")
        print("-" * 55)

        for threshold_name, current_threshold in [
            ("Default 0.50", 0.50),
            ("Validation-selected", threshold)
        ]:

            predictions = (
                probabilities >= current_threshold
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1]
            ).ravel()

            false_positive_rate = fp / (fp + tn)

            print(f"\n{threshold_name}")
            print(f"Threshold: {current_threshold:.6f}")
            print(f"Accuracy: {accuracy_score(y_true, predictions):.4f}")
            print(f"Attack precision: {precision_score(y_true, predictions, zero_division=0):.4f}")
            print(f"Attack recall: {recall_score(y_true, predictions, zero_division=0):.4f}")
            print(f"Attack F1: {f1_score(y_true, predictions, zero_division=0):.4f}")
            print(f"False positive rate: {false_positive_rate:.4f}")
            print(f"False positives: {fp}")
            print(f"Missed attacks: {fn}")

print("\nTHRESHOLD ANALYSIS COMPLETED")
