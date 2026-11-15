
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DATA_FILE = Path("data/processed/model_ready_dataset.csv")
MODEL_FILE = Path("models/hist_gradient_boosting_family_aware.joblib")
RESULT_DIR = Path("results")

CHUNK_SIZE = 100_000

print("=" * 70)
print("FRIDAY ATTACK PROBABILITY ANALYSIS")
print("=" * 70)

saved = joblib.load(MODEL_FILE)
pipeline = saved["pipeline"]
features = saved["feature_columns"]

parts = []

for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
    friday_attacks = chunk[
        (chunk["Source_Day"].astype(str).str.strip() == "Friday")
        & (chunk["Target"] == 1)
    ]

    if friday_attacks.empty:
        continue

    probabilities = pipeline.predict_proba(
        friday_attacks[features]
    )[:, 1]

    result = pd.DataFrame({
        "Attack_Family": friday_attacks["Label"].astype(str).values,
        "Attack_Probability": probabilities
    })

    parts.append(result)

    print(
        "Attack records processed:",
        f"{sum(len(part) for part in parts):,}",
        end="\r"
    )

data = pd.concat(parts, ignore_index=True)

print("\n\nTotal Friday attacks:", f"{len(data):,}")

summary = data.groupby("Attack_Family")["Attack_Probability"].agg(
    Total_Attacks="count",
    Mean_Probability="mean",
    Median_Probability="median",
    Minimum_Probability="min",
    Maximum_Probability="max"
)

summary["Detected_At_0.50"] = data.groupby(
    "Attack_Family"
)["Attack_Probability"].apply(
    lambda values: int((values >= 0.50).sum())
)

summary["Recall_At_0.50"] = (
    summary["Detected_At_0.50"] / summary["Total_Attacks"]
)

summary = summary.sort_values("Mean_Probability", ascending=False)

print("\nATTACK FAMILY PROBABILITY SUMMARY")
print("=" * 70)
print(summary.to_string())

output = RESULT_DIR / "friday_attack_probability_summary.csv"
summary.to_csv(output)

print("\nSaved:", output)

print("\nPROBABILITY DISTRIBUTION BY ATTACK FAMILY")
print("=" * 70)

print(
    data.groupby("Attack_Family")["Attack_Probability"]
    .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    .unstack()
    .to_string()
)

print("\nANALYSIS COMPLETED")
