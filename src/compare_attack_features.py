
import pandas as pd
import numpy as np

DATA_PATH = "data/processed/model_ready_dataset.csv"

print("=" * 70)
print("ATTACK FEATURE DISTRIBUTION COMPARISON")
print("=" * 70)

# Features that may help distinguish network traffic patterns.
selected_features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "RST Flag Count",
    "Average Packet Size"
]

# Read training attack records.
train = pd.read_csv(
    "data/processed/train.csv",
    usecols=lambda c: c in selected_features + ["Target"]
)

train_attack = train[train["Target"] == 1]

# Read Friday attack records.
friday = pd.read_csv(
    DATA_PATH,
    usecols=lambda c: c in selected_features + [
        "Target", "Label", "Source_Day"
    ]
)

friday_attack = friday[
    (friday["Source_Day"] == "Friday") &
    (friday["Target"] == 1)
]

groups = {
    "Training_Attacks": train_attack,
    "Friday_DDoS": friday_attack[
        friday_attack["Label"] == "DDoS"
    ],
    "Friday_PortScan": friday_attack[
        friday_attack["Label"] == "PortScan"
    ],
    "Friday_Bot": friday_attack[
        friday_attack["Label"] == "Bot"
    ]
}

for feature in selected_features:
    print("\n" + "=" * 70)
    print("FEATURE:", feature)
    print("-" * 70)

    for group_name, data in groups.items():
        values = pd.to_numeric(
            data[feature],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if len(values) == 0:
            print(group_name, ": No valid values")
            continue

        print(
            f"{group_name}: "
            f"count={len(values)}, "
            f"median={values.median():.4f}, "
            f"mean={values.mean():.4f}, "
            f"25%={values.quantile(0.25):.4f}, "
            f"75%={values.quantile(0.75):.4f}"
        )

print("\nFEATURE COMPARISON COMPLETED")
