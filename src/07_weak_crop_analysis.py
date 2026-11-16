
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
WEAK_OUTPUT_DIR = OUTPUT_DIR / "weak_crop_analysis"

# ==========================================
# 1. LOAD TEST DATA AND PREDICTIONS
# ==========================================

X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv")

predictions = pd.read_csv(
    OUTPUT_DIR / "model_predictions.csv"
)

# Identify Random Forest predictions
actual = predictions["Actual_Yield"]
predicted = predictions["Random Forest Regressor"]

# Reconstruct crop names from one-hot columns
crop_columns = [
    col for col in X_test.columns
    if col.startswith("Crop_")
]

if not crop_columns:
    raise ValueError("Crop columns not found!")

crop_names = (
    X_test[crop_columns]
    .idxmax(axis=1)
    .str.replace("Crop_", "", regex=False)
)

# Create analysis dataframe
analysis_df = pd.DataFrame({
    "Crop": crop_names,
    "Actual_Yield": actual,
    "Predicted_Yield": predicted
})

analysis_df["Residual"] = (
    analysis_df["Actual_Yield"]
    - analysis_df["Predicted_Yield"]
)

# Output directory
WEAK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================
# 2. SELECT WEAK-PERFORMING CROPS
# ==========================================

weak_crops = [
    "Chickpea",
    "Pigeonpea",
    "Cotton",
    "Soybean"
]


# ==========================================
# 3. ACTUAL VS PREDICTED GRAPHS
# ==========================================

for crop in weak_crops:

    crop_data = analysis_df[
        analysis_df["Crop"] == crop
    ]

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=crop_data,
        x="Actual_Yield",
        y="Predicted_Yield",
        alpha=0.5
    )

    # Perfect prediction line
    minimum = min(
        crop_data["Actual_Yield"].min(),
        crop_data["Predicted_Yield"].min()
    )

    maximum = max(
        crop_data["Actual_Yield"].max(),
        crop_data["Predicted_Yield"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect Prediction"
    )

    plt.title(f"{crop}: Actual vs Predicted Yield")
    plt.xlabel("Actual Yield (kg/ha)")
    plt.ylabel("Predicted Yield (kg/ha)")
    plt.legend()

    plt.tight_layout()

    filename = (
        crop.lower().replace(" ", "_")
        + "_actual_vs_predicted.png"
    )

    plt.savefig(WEAK_OUTPUT_DIR / filename)

    plt.show()


# ==========================================
# 4. CROP-WISE RESIDUAL DISTRIBUTION
# ==========================================

for crop in weak_crops:

    crop_data = analysis_df[
        analysis_df["Crop"] == crop
    ]

    plt.figure(figsize=(9, 5))

    sns.histplot(
        crop_data["Residual"],
        bins=30,
        kde=True
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Zero Error"
    )

    plt.title(f"{crop}: Prediction Error Distribution")
    plt.xlabel("Residual (Actual - Predicted) in kg/ha")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()

    filename = (
        crop.lower().replace(" ", "_")
        + "_residual_distribution.png"
    )

    plt.savefig(WEAK_OUTPUT_DIR / filename)

    plt.show()


# ==========================================
# 5. PRINT CROP-WISE SUMMARY
# ==========================================

print("\nWEAK CROP ANALYSIS SUMMARY")

for crop in weak_crops:

    crop_data = analysis_df[
        analysis_df["Crop"] == crop
    ]

    print(f"\nCrop: {crop}")
    print("Records:", len(crop_data))

    print(
        "Average Actual Yield:",
        round(crop_data["Actual_Yield"].mean(), 2)
    )

    print(
        "Average Predicted Yield:",
        round(crop_data["Predicted_Yield"].mean(), 2)
    )

    print(
        "Average Residual:",
        round(crop_data["Residual"].mean(), 2)
    )

    print(
        "MAE:",
        round(crop_data["Residual"].abs().mean(), 2)
    )

print("\nWEAK CROP ANALYSIS COMPLETED")
