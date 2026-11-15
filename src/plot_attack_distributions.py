
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/processed/model_ready_dataset.csv"

features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Average Packet Size"
]

print("Loading training attacks...")

train = pd.read_csv("data/processed/train.csv")

train_attack = train[train["Target"] == 1].copy()
train_attack["Group"] = "Training attacks"

del train

print("Loading Friday attacks...")

friday = pd.read_csv(DATA_PATH)

friday = friday[
    (friday["Source_Day"] == "Friday") &
    (friday["Target"] == 1)
].copy()

friday["Group"] = friday["Label"]

# Use a reproducible sample to keep plots manageable.
samples = [train_attack.sample(
    n=min(10000, len(train_attack)),
    random_state=42
)]

for label in ["DDoS", "PortScan", "Bot"]:
    subset = friday[friday["Label"] == label]

    if len(subset) > 0:
        samples.append(
            subset.sample(
                n=min(10000, len(subset)),
                random_state=42
            )
        )

plot_data = pd.concat(samples, ignore_index=True)

groups = [
    "Training attacks",
    "DDoS",
    "PortScan",
    "Bot"
]

for feature in features:

    plot_data[feature] = pd.to_numeric(
        plot_data[feature],
        errors="coerce"
    )

    # Log transformation handles highly skewed positive values.
    plot_data["Log_" + feature] = np.log1p(
        plot_data[feature].clip(lower=0)
    )

    plt.figure(figsize=(11, 6))

    sns.boxplot(
        data=plot_data,
        x="Group",
        y="Log_" + feature,
        order=groups
    )

    plt.title(
        f"Log-Transformed Distribution: {feature}"
    )

    plt.xlabel("Attack Group")
    plt.ylabel("log(1 + feature value)")
    plt.xticks(rotation=15)
    plt.tight_layout()

    filename = (
        "data/processed/"
        + feature.replace("/", "_").replace(" ", "_")
        + "_distribution.png"
    )

    plt.savefig(filename, dpi=150)
    plt.show()
    plt.close()

print("All distribution plots generated.")

