
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import confusion_matrix

INPUT_FILE = Path("data/processed/model_ready_dataset.csv")
MODEL_FILE = Path("models/random_forest_baseline.joblib")

CHUNK_SIZE = 100_000

print("=" * 65)
print("FRIDAY ATTACK ERROR ANALYSIS")
print("=" * 65)

saved = joblib.load(MODEL_FILE)
pipeline = saved["pipeline"]
features = saved["feature_columns"]

results = []

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):

    friday = chunk[chunk["Source_Day"] == "Friday"]

    if friday.empty:
        continue

    X = friday[features]

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "Label": friday["Label"].values,
        "Actual": friday["Target"].values,
        "Predicted": predictions,
        "Attack_Probability": probabilities
    })

    results.append(result)

    print("Processed Friday records:", sum(len(x) for x in results))

df = pd.concat(results, ignore_index=True)

# --------------------------------------------------
# ATTACK-LABEL ANALYSIS
# --------------------------------------------------

print("\nATTACK DETECTION BY ORIGINAL LABEL")
print("-" * 65)

attacks = df[df["Actual"] == 1]

for label, group in attacks.groupby("Label"):

    detected = (group["Predicted"] == 1).sum()
    missed = (group["Predicted"] == 0).sum()

    recall = detected / len(group)

    print("\nAttack:", label)
    print("Total:", f"{len(group):,}")
    print("Detected:", f"{detected:,}")
    print("Missed:", f"{missed:,}")
    print("Recall:", f"{recall:.4%}")
    print(
        "Median attack probability:",
        round(group["Attack_Probability"].median(), 4)
    )

# --------------------------------------------------
# PROBABILITY ANALYSIS
# --------------------------------------------------

print("\nATTACK PROBABILITY DISTRIBUTION")
print("-" * 65)

for label, group in attacks.groupby("Label"):

    print("\n", label)

    print(
        group["Attack_Probability"].describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
        ).round(4)
    )

# --------------------------------------------------
# DETECTION THRESHOLD DIAGNOSTIC
# --------------------------------------------------

print("\nTHRESHOLD DIAGNOSTIC")
print("-" * 65)

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:

    predicted = (df["Attack_Probability"] >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        df["Actual"],
        predicted,
        labels=[0, 1]
    ).ravel()

    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Attack Recall: {recall:.4f} | "
        f"False Positive Rate: {fpr:.4f}"
    )

print("\nANALYSIS COMPLETED")
