
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/random_forest_baseline.joblib"

print("=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

# 1. Load the saved model
saved = joblib.load(MODEL_PATH)

pipeline = saved["pipeline"]
feature_columns = saved["feature_columns"]

# 2. Automatically find Random Forest inside the pipeline
rf = None

for step_name, step_object in pipeline.named_steps.items():

    print(f"Pipeline step: {step_name} -> {type(step_object).__name__}")

    if isinstance(step_object, RandomForestClassifier):
        rf = step_object
        break

if rf is None:
    raise ValueError("Random Forest model not found inside the pipeline.")

# 3. Extract feature importance
importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": rf.feature_importances_
})

# 4. Sort by importance
importance = importance.sort_values(
    by="Importance",
    ascending=False
)

# 5. Display top 20 features
print("\nTOP 20 IMPORTANT FEATURES")
print("-" * 60)

print(
    importance.head(20).to_string(index=False)
)

# 6. Save complete feature importance
importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

# 7. Plot top 20 features
top20 = importance.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(12, 8))

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Network Feature")
plt.title("Top 20 Random Forest Feature Importances")

plt.tight_layout()

plt.savefig(
    "data/processed/feature_importance.png",
    dpi=150
)

plt.show()
plt.close()

print("\nFeature importance analysis completed.")
print("CSV saved: data/processed/feature_importance.csv")
print("Graph saved: data/processed/feature_importance.png")
