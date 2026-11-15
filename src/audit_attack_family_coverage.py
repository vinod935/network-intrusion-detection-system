
import pandas as pd

print("=" * 70)
print("ATTACK FAMILY COVERAGE AUDIT")
print("=" * 70)

df = pd.read_csv(
    "data/processed/model_ready_dataset.csv",
    usecols=["Label", "Target", "Source_Day"]
)

# Training development days
training = df[df["Source_Day"] != "Friday"]

# Friday diagnostic data
friday = df[df["Source_Day"] == "Friday"]

training_labels = set(training["Label"].unique())
friday_labels = set(friday["Label"].unique())

print("\n1. ATTACK LABELS IN TRAINING DAYS")
print("-" * 70)

training_counts = training["Label"].value_counts()

for label, count in training_counts.items():
    print(f"{label:<40} {count:>12,}")

print("\n2. ATTACK LABELS IN FRIDAY")
print("-" * 70)

friday_counts = friday["Label"].value_counts()

for label, count in friday_counts.items():
    print(f"{label:<40} {count:>12,}")

print("\n3. FRIDAY LABEL COVERAGE")
print("-" * 70)

for label in sorted(friday_labels):

    if label in training_labels:
        status = "PRESENT IN TRAINING"
    else:
        status = "NOT PRESENT IN TRAINING"

    print(f"{label:<40} {status}")

print("\n4. SUMMARY")
print("-" * 70)

missing = friday_labels - training_labels

print("Training labels:", len(training_labels))
print("Friday labels:", len(friday_labels))
print("Friday labels absent from training:", len(missing))

print("\nMissing Friday labels:")

for label in sorted(missing):
    print("-", label)

print("\nAUDIT COMPLETED")
