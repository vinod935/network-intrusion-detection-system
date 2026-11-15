import pandas as pd
import numpy as np
import joblib
import time

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

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
# PATHS
# --------------------------------------------------

TRAIN_FILE = Path("data/processed/train_grouped.csv")
VAL_FILE = Path("data/processed/validation_grouped.csv")

MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 70)
print("GROUP-BASED MODEL TRAINING")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

excluded = ["Target", "Label", "Source_Day", "Source_File"]

feature_columns = [
    col for col in train_df.columns
    if col not in excluded
]

X_train = train_df[feature_columns]
y_train = train_df["Target"]

X_val = val_df[feature_columns]
y_val = val_df["Target"]

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Features:", len(feature_columns))

if len(feature_columns) != 69:
    raise ValueError(
        f"Expected 69 features, found {len(feature_columns)}"
    )

if list(X_train.columns) != list(X_val.columns):
    raise ValueError("Training and validation features do not match.")

# --------------------------------------------------
# MODEL DEFINITIONS
# --------------------------------------------------

models = {
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=4,
            max_features="sqrt"
        ))
    ]),

    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=4,
            max_features="sqrt"
        ))
    ]),

    "Hist Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.1,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42
        ))
    ])
}

# --------------------------------------------------
# TRAIN AND EVALUATE
# --------------------------------------------------

results = []

for name, pipeline in models.items():

    print("\n" + "=" * 70)
    print("MODEL:", name)
    print("=" * 70)

    start = time.time()

    pipeline.fit(X_train, y_train)

    training_time = time.time() - start

    predictions = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

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
    roc_auc = roc_auc_score(y_val, probabilities)
    pr_auc = average_precision_score(y_val, probabilities)

    cm = confusion_matrix(y_val, predictions, labels=[0, 1])

    print(f"\nTraining time: {training_time:.2f} seconds")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_val,
        predictions,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        digits=4,
        zero_division=0
    ))

    print("CONFUSION MATRIX")
    print(cm)

    print("\nMETRICS")
    print("Accuracy:", round(accuracy, 4))
    print("Attack Precision:", round(precision, 4))
    print("Attack Recall:", round(recall, 4))
    print("Attack F1:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    filename = name.lower().replace(" ", "_") + "_grouped.joblib"
    model_path = MODEL_DIR / filename

    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": feature_columns
    }, model_path)

    print("\nModel saved:", model_path)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Attack_Precision": precision,
        "Attack_Recall": recall,
        "Attack_F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Training_Time_Seconds": training_time,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    })

# --------------------------------------------------
# SAVE COMPARISON
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = RESULT_DIR / "grouped_model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 70)
print("FINAL GROUPED MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))

print("\nComparison saved:", results_path)
print("\nALL GROUPED MODELS TRAINED SUCCESSFULLY")

