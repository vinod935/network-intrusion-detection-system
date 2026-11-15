import pandas as pd
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

TRAIN_FILE = Path("data/processed/train_family_aware.csv")
VAL_FILE = Path("data/processed/validation_family_aware.csv")

MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("FAMILY-AWARE MODEL TRAINING")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

excluded = ["Target", "Label", "Source_Day", "Source_File"]
feature_columns = [c for c in train_df.columns if c not in excluded]

X_train = train_df[feature_columns]
y_train = train_df["Target"]

X_val = val_df[feature_columns]
y_val = val_df["Target"]

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Features:", len(feature_columns))

if len(feature_columns) != 69:
    raise ValueError("Expected 69 features.")

if list(X_train.columns) != list(X_val.columns):
    raise ValueError("Feature columns do not match.")

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

results = []
family_results = []

for name, pipeline in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", name)
    print("=" * 70)

    start = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start

    predictions = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_val, probabilities)
    pr_auc = average_precision_score(y_val, probabilities)

    cm = confusion_matrix(y_val, predictions, labels=[0, 1])

    print("Training time:", round(training_time, 2), "seconds")

    print("\nClassification report:")
    print(classification_report(
        y_val,
        predictions,
        target_names=["BENIGN", "ATTACK"],
        digits=4,
        zero_division=0
    ))

    print("Confusion matrix:")
    print(cm)

    print("\nAttack precision:", round(precision, 4))
    print("Attack recall:", round(recall, 4))
    print("Attack F1:", round(f1, 4))
    print("Accuracy:", round(accuracy, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    family_data = val_df[["Label", "Target"]].copy()
    family_data["Prediction"] = predictions

    print("\nAttack-family recall:")

    for family, group in family_data[
        family_data["Target"] == 1
    ].groupby("Label"):

        support = len(group)
        detected = int((group["Prediction"] == 1).sum())
        family_recall = detected / support

        print(
            f"{family}: {detected}/{support}, "
            f"recall={family_recall:.4f}"
        )

        family_results.append({
            "Model": name,
            "Attack_Family": family,
            "Support": support,
            "Detected": detected,
            "Missed": support - detected,
            "Recall": family_recall
        })

    filename = name.lower().replace(" ", "_") + "_family_aware.joblib"
    model_path = MODEL_DIR / filename

    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": feature_columns
    }, model_path)

    print("Model saved:", model_path)

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

pd.DataFrame(results).to_csv(
    RESULT_DIR / "family_aware_model_comparison.csv",
    index=False
)

pd.DataFrame(family_results).to_csv(
    RESULT_DIR / "family_aware_attack_family_recall.csv",
    index=False
)

print("\nALL FAMILY-AWARE MODELS TRAINED SUCCESSFULLY")
