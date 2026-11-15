
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

MODEL_PATH = "models/random_forest_baseline.joblib"
VALIDATION_PATH = "data/processed/validation.csv"

print("=" * 65)
print("PERMUTATION FEATURE IMPORTANCE")
print("=" * 65)

# Load saved pipeline
saved = joblib.load(MODEL_PATH)

pipeline = saved["pipeline"]
feature_columns = saved["feature_columns"]

# Load validation data
validation = pd.read_csv(VALIDATION_PATH)

X_val = validation[feature_columns]
y_val = validation["Target"]

print("\nValidation records:", len(X_val))
print("Features:", len(feature_columns))

# Calculate permutation importance
print("\nCalculating permutation importance...")
print("This may take some time.")

result = permutation_importance(
    pipeline,
    X_val,
    y_val,
    scoring="f1",
    n_repeats=3,
    random_state=42,
    n_jobs=4
)

# Create results table
importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance_Mean": result.importances_mean,
    "Importance_Std": result.importances_std
})

importance = importance.sort_values(
    by="Importance_Mean",
    ascending=False
)

print("\nTOP 20 PERMUTATION FEATURES")
print("-" * 65)

print(
    importance.head(20).to_string(index=False)
)

# Save complete results
importance.to_csv(
    "data/processed/permutation_feature_importance.csv",
    index=False
)

# Plot top 20
top20 = importance.head(20).sort_values(
    by="Importance_Mean",
    ascending=True
)

plt.figure(figsize=(12, 8))

plt.barh(
    top20["Feature"],
    top20["Importance_Mean"],
    xerr=top20["Importance_Std"]
)

plt.xlabel("Decrease in Validation F1 after Shuffling")
plt.ylabel("Network Feature")
plt.title("Permutation Feature Importance")

plt.tight_layout()

plt.savefig(
    "data/processed/permutation_feature_importance.png",
    dpi=150
)

plt.show()
plt.close()

print("\nPermutation importance completed.")
