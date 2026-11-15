
from pathlib import Path

import joblib
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "hist_gradient_boosting_family_aware.joblib"


class NIDSPredictor:

    def __init__(self):
        # Load saved model dictionary
        saved_model = joblib.load(MODEL_PATH)

        self.pipeline = saved_model["pipeline"]
        self.feature_columns = saved_model["feature_columns"]

        print("NIDS model loaded successfully")
        print(f"Expected features: {len(self.feature_columns)}")

    def predict(self, input_csv):

        # Read network flow data
        data = pd.read_csv(input_csv)

        # Check required features
        missing_features = [
            col for col in self.feature_columns
            if col not in data.columns
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        # Select features in exact training order
        X = data[self.feature_columns].copy()

        # Convert features to numeric
        X = X.apply(pd.to_numeric, errors="coerce")

        # Generate predictions
        predictions = self.pipeline.predict(X)

        probabilities = self.pipeline.predict_proba(X)

        # Build results
        results = pd.DataFrame({
            "Prediction": predictions,
            "Attack_Probability": probabilities[:, 1]
        })

        results["Status"] = results["Prediction"].map({
            0: "BENIGN",
            1: "ATTACK"
        })

        return results


if __name__ == "__main__":
    predictor = NIDSPredictor()

    print("Prediction engine initialized successfully.")
