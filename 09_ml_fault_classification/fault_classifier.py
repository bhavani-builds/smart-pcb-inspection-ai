import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ------------------------------------------
# Project paths
# ------------------------------------------

project_folder = (
    Path(__file__).resolve().parent.parent
)

features_file = (
    project_folder
    / "08_feature_extraction"
    / "fault_features.csv"
)


# ------------------------------------------
# Load extracted features
# ------------------------------------------

data = pd.read_csv(
    features_file
)


if data.empty:

    print("❌ No feature data found.")

    raise SystemExit


print("=" * 65)
print("             PCB FAULT CLASSIFICATION")
print("=" * 65)

print(
    f"\nFault samples available: {len(data)}"
)


# ------------------------------------------
# Create labels
# ------------------------------------------

# For our prototype, assign labels based
# on the simulated fault regions.
#
# This is NOT a real-world labeling method.
# Real PCB inspection requires manually
# verified defect labels.

labels = []

for index in range(len(data)):

    if index % 2 == 0:

        labels.append(
            "BROKEN_TRACE"
        )

    else:

        labels.append(
            "DAMAGE"
        )


data["fault_type"] = labels


# ------------------------------------------
# Features
# ------------------------------------------

feature_columns = [
    "width",
    "height",
    "area",
    "aspect_ratio",
    "average_brightness",
    "edge_density"
]


X = data[
    feature_columns
]

y = data[
    "fault_type"
]


# ------------------------------------------
# Check dataset size
# ------------------------------------------

if len(data) < 4:

    print(
        "\n⚠️ The current simulated dataset "
        "contains too few samples for a meaningful "
        "train/test split."
    )

    print(
        "\nFor now, the feature data is valid, "
        "but collect more labeled fault samples "
        "before treating the ML accuracy as meaningful."
    )

    raise SystemExit


# ------------------------------------------
# Split dataset
# ------------------------------------------

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )
)


# ------------------------------------------
# Create model
# ------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ------------------------------------------
# Train model
# ------------------------------------------

model.fit(
    X_train,
    y_train
)


# ------------------------------------------
# Predictions
# ------------------------------------------

predictions = model.predict(
    X_test
)


# ------------------------------------------
# Accuracy
# ------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    f"\nModel Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ------------------------------------------
# Classification report
# ------------------------------------------

print(
    "\nClassification Report"
)

print(
    "-" * 50
)

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ------------------------------------------
# Feature importance
# ------------------------------------------

importance = model.feature_importances_

feature_importance = pd.DataFrame({

    "Feature":
        feature_columns,

    "Importance":
        importance

})


feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print(
    "\nFeature Importance"
)

print(
    "-" * 50
)

print(
    feature_importance.to_string(
        index=False
    )
)


# ------------------------------------------
# Visualization
# ------------------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title(
    "PCB Fault Classification — Feature Importance",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Feature"
)

plt.ylabel(
    "Importance"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()


# ------------------------------------------
# Save visualization
# ------------------------------------------

output_folder = (
    project_folder
    / "09_ml_fault_classification"
)

output_folder.mkdir(
    exist_ok=True
)


figure_path = (
    output_folder
    / "Figure_9.png"
)


plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------
# Save feature importance
# ------------------------------------------

importance_file = (
    output_folder
    / "feature_importance.csv"
)


feature_importance.to_csv(
    importance_file,
    index=False
)


print(
    "\nFeature importance saved to:"
)

print(
    importance_file
)

print(
    "\nVisualization saved to:"
)

print(
    figure_path
)
