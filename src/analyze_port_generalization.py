
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/random_forest_baseline.joblib"
TRAIN_PATH = "data/processed/train.csv"
DATA_PATH = "data/processed/model_ready_dataset.csv"

print("=" * 70)
print("DESTINATION PORT GENERALIZATION ANALYSIS")
print("=" * 70)

# Load trained pipeline
saved = joblib.load(MODEL_PATH)
pipeline = saved["pipeline"]
features = saved["feature_columns"]

# --------------------------------------------------
# 1. Training attack distribution by destination port
# --------------------------------------------------

print("\n1. TRAINING ATTACK DISTRIBUTION")
print("-" * 70)

train = pd.read_csv(TRAIN_PATH)

train_attacks = train[train["Target"] == 1]

train_ports = (
    train_attacks.groupby("Destination Port")
    .size()
    .sort_values(ascending=False)
    .head(20)
)

print(train_ports.to_string())

del train

# --------------------------------------------------
# 2. Evaluate Friday by destination port
# --------------------------------------------------

print("\n2. FRIDAY ATTACK DETECTION BY DESTINATION PORT")
print("-" * 70)

results = []

chunksize = 100000

for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize):

    friday = chunk[chunk["Source_Day"] == "Friday"].copy()

    if friday.empty:
        continue

    X = friday[features]
    y = friday["Target"]

    predictions = pipeline.predict(X)

    friday["Predicted"] = predictions

    results.append(
        friday[
            ["Destination Port", "Target", "Predicted"]
        ]
    )

    print("Processed Friday records:", sum(len(x) for x in results))

friday_data = pd.concat(results, ignore_index=True)

# --------------------------------------------------
# 3. Calculate metrics for each destination port
# --------------------------------------------------

summary = (
    friday_data.groupby("Destination Port")
    .apply(
        lambda group: pd.Series({
            "Total": len(group),
            "Actual_Attacks": int((group["Target"] == 1).sum()),
            "Detected_Attacks": int(
                ((group["Target"] == 1) &
                 (group["Predicted"] == 1)).sum()
            )
        }),
        include_groups=False
    )
    .reset_index()
)

summary["Attack_Recall"] = np.where(
    summary["Actual_Attacks"] > 0,
    summary["Detected_Attacks"] / summary["Actual_Attacks"],
    np.nan
)

summary = summary.sort_values(
    by="Actual_Attacks",
    ascending=False
)

print("\nTOP 30 FRIDAY DESTINATION PORTS")
print("-" * 70)

print(summary.head(30).to_string(index=False))

# Save results
summary.to_csv(
    "data/processed/friday_port_detection.csv",
    index=False
)

print("\nSaved: data/processed/friday_port_detection.csv")
print("\nPORT GENERALIZATION ANALYSIS COMPLETED")
