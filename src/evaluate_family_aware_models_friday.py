import pandas as pd
import joblib
import time

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = Path("data/processed/model_ready_dataset.csv")

CHUNK_SIZE = 100_000
HELD_OUT_DAY = "Friday"

MODEL_FILES = {
    "Random Forest": "models/random_forest_family_aware.joblib",
    "Extra Trees": "models/extra_trees_family_aware.joblib",
    "Hist Gradient Boosting": "models/hist_gradient_boosting_family_aware.joblib"
}

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

summary_results = []
attack_family_results = []
source_results = []

# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(model_name, model_path):

    print("\n" + "=" * 75)
    print("MODEL:", model_name)
    print("=" * 75)

    saved = joblib.load(model_path)

    pipeline = saved["pipeline"]
    feature_columns = saved["feature_columns"]

    print("Model loaded:", model_path)
    print("Features:", len(feature_columns))

    if len(feature_columns) != 69:
        raise ValueError(
            f"{model_name}: Expected 69 features, "
            f"found {len(feature_columns)}"
        )

    actual_all = []
    predicted_all = []
    probability_all = []
    labels_all = []
    sources_all = []

    total_records = 0

    start_time = time.time()

    print("\nReading Friday records in chunks...")

    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):

        friday = chunk[
            chunk["Source_Day"].astype(str).str.strip()
            == HELD_OUT_DAY
        ]

        if friday.empty:
            continue

        X = friday[feature_columns]
        y = friday["Target"]

        predictions = pipeline.predict(X)
        probabilities = pipeline.predict_proba(X)[:, 1]

        actual_all.extend(y.tolist())
        predicted_all.extend(predictions.tolist())
        probability_all.extend(probabilities.tolist())

        labels_all.extend(friday["Label"].astype(str).tolist())
        sources_all.extend(friday["Source_File"].astype(str).tolist())

        total_records += len(friday)

        print(
            f"Records evaluated: {total_records:,}",
            end="\r"
        )

    elapsed = time.time() - start_time

    actual = pd.Series(actual_all)
    predicted = pd.Series(predicted_all)
    probabilities = pd.Series(probability_all)

    labels = pd.Series(labels_all)
    sources = pd.Series(sources_all)

    if actual.empty:
        raise ValueError("No Friday records found.")

    # --------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual, predicted, zero_division=0
    )

    recall = recall_score(
        actual, predicted, zero_division=0
    )

    f1 = f1_score(
        actual, predicted, zero_division=0
    )

    macro_f1 = f1_score(
        actual, predicted, average="macro"
    )

    roc_auc = roc_auc_score(actual, probabilities)

    pr_auc = average_precision_score(
        actual, probabilities
    )

    cm = confusion_matrix(
        actual, predicted, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n\nFRIDAY RECORDS:", f"{total_records:,}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 75)

    print(classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        digits=4,
        zero_division=0
    ))

    print("CONFUSION MATRIX")
    print(cm)

    print("\nOVERALL METRICS")
    print("Accuracy:", round(accuracy, 6))
    print("Attack Precision:", round(precision, 6))
    print("Attack Recall:", round(recall, 6))
    print("Attack F1:", round(f1, 6))
    print("Macro F1:", round(macro_f1, 6))
    print("ROC-AUC:", round(roc_auc, 6))
    print("PR-AUC:", round(pr_auc, 6))
    print("Evaluation time:", round(elapsed, 2), "seconds")

    summary_results.append({
        "Model": model_name,
        "Records": total_records,
        "Accuracy": accuracy,
        "Attack_Precision": precision,
        "Attack_Recall": recall,
        "Attack_F1": f1,
        "Macro_F1": macro_f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Evaluation_Time_Seconds": elapsed
    })

    # --------------------------------------------------
    # ATTACK FAMILY ANALYSIS
    # --------------------------------------------------

    results = pd.DataFrame({
        "Label": labels,
        "Actual": actual,
        "Predicted": predicted
    })

    print("\nATTACK DETECTION BY ORIGINAL LABEL")
    print("-" * 75)

    for label, group in results.groupby("Label"):

        attack_rows = group[group["Actual"] == 1]

        if attack_rows.empty:
            continue

        total = len(attack_rows)

        detected = (
            attack_rows["Predicted"] == 1
        ).sum()

        missed = total - detected

        family_recall = detected / total

        print(
            f"{label}: "
            f"Total={total:,}, "
            f"Detected={detected:,}, "
            f"Missed={missed:,}, "
            f"Recall={family_recall:.4%}"
        )

        attack_family_results.append({
            "Model": model_name,
            "Attack_Family": label,
            "Total_Attacks": total,
            "Detected": detected,
            "Missed": missed,
            "Recall": family_recall
        })

    # --------------------------------------------------
    # SOURCE FILE ANALYSIS
    # --------------------------------------------------

    source_df = pd.DataFrame({
        "Source_File": sources,
        "Actual": actual,
        "Predicted": predicted
    })

    print("\nPER-FILE PERFORMANCE")
    print("-" * 75)

    for filename, group in source_df.groupby("Source_File"):

        benign = group[group["Actual"] == 0]
        attacks = group[group["Actual"] == 1]

        file_accuracy = (
            (group["Actual"] == group["Predicted"]).mean()
        )

        attack_recall = (
            (attacks["Predicted"] == 1).mean()
            if len(attacks) else None
        )

        false_positive_rate = (
            (benign["Predicted"] == 1).mean()
            if len(benign) else None
        )

        print("\nFile:", filename)
        print("Records:", len(group))
        print("Accuracy:", round(file_accuracy, 4))
        print(
            "Attack Recall:",
            round(attack_recall, 4)
            if attack_recall is not None else "N/A"
        )
        print(
            "False Positive Rate:",
            round(false_positive_rate, 4)
            if false_positive_rate is not None else "N/A"
        )

        source_results.append({
            "Model": model_name,
            "Source_File": filename,
            "Records": len(group),
            "Accuracy": file_accuracy,
            "Attack_Recall": attack_recall,
            "False_Positive_Rate": false_positive_rate
        })


# --------------------------------------------------
# RUN ALL THREE MODELS
# --------------------------------------------------

print("=" * 75)
print("FAMILY-AWARE MODELS - UNSEEN FRIDAY EVALUATION")
print("=" * 75)

for model_name, model_path in MODEL_FILES.items():

    evaluate_model(model_name, model_path)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

summary_df = pd.DataFrame(summary_results)

family_df = pd.DataFrame(attack_family_results)

source_df = pd.DataFrame(source_results)

summary_df.to_csv(
    RESULT_DIR / "family_aware_friday_model_comparison.csv",
    index=False
)

family_df.to_csv(
    RESULT_DIR / "family_aware_friday_attack_family_recall.csv",
    index=False
)

source_df.to_csv(
    RESULT_DIR / "family_aware_friday_source_analysis.csv",
    index=False
)

print("\n" + "=" * 75)
print("FINAL FRIDAY MODEL COMPARISON")
print("=" * 75)

print(summary_df.to_string(index=False))

print("\nResults saved:")
print("1. results/family_aware_friday_model_comparison.csv")
print("2. results/family_aware_friday_attack_family_recall.csv")
print("3. results/family_aware_friday_source_analysis.csv")

print("\nFAMILY-AWARE FRIDAY EVALUATION COMPLETED")

