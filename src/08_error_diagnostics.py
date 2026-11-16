
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DIAGNOSTICS_DIR = OUTPUT_DIR / "error_diagnostics"

# ==========================================
# 1. LOAD DATA
# ==========================================

X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv")
X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv")

y_train = pd.read_csv(OUTPUT_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(OUTPUT_DIR / "y_test.csv").squeeze()

predictions = pd.read_csv(
    OUTPUT_DIR / "model_predictions.csv"
)

# ==========================================
# 2. IDENTIFY CROP NAMES
# ==========================================

crop_columns = [
    col for col in X_test.columns
    if col.startswith("Crop_")
]

test_crops = (
    X_test[crop_columns]
    .idxmax(axis=1)
    .str.replace("Crop_", "", regex=False)
)

train_crops = (
    X_train[crop_columns]
    .idxmax(axis=1)
    .str.replace("Crop_", "", regex=False)
)

# ==========================================
# 3. CREATE TEST ANALYSIS DATAFRAME
# ==========================================

analysis = pd.DataFrame({
    "Crop": test_crops,
    "Year": X_test["Year"],
    "Actual_Yield": y_test,
    "Predicted_Yield": predictions[
        "Random Forest Regressor"
    ]
})

analysis["Residual"] = (
    analysis["Actual_Yield"]
    - analysis["Predicted_Yield"]
)

analysis["Absolute_Error"] = (
    analysis["Residual"].abs()
)

weak_crops = [
    "Chickpea",
    "Pigeonpea",
    "Cotton",
    "Soybean"
]

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 4. FIND TOP 10 LARGEST ERRORS
# ==========================================

for crop in weak_crops:

    crop_data = analysis[
        analysis["Crop"] == crop
    ]

    top_errors = crop_data.nlargest(
        10,
        "Absolute_Error"
    )

    print("\n================================")
    print("CROP:", crop)
    print("TOP 10 LARGEST PREDICTION ERRORS")
    print("================================")

    print(
        top_errors[
            [
                "Year",
                "Actual_Yield",
                "Predicted_Yield",
                "Residual",
                "Absolute_Error"
            ]
        ].to_string(index=False)
    )

    filename = (
        crop.lower().replace(" ", "_")
        + "_top_errors.csv"
    )

    top_errors.to_csv(
        DIAGNOSTICS_DIR / filename,
        index=False
    )


# ==========================================
# 5. COMPARE TRAINING AND TESTING YIELDS
# ==========================================

train_summary = pd.DataFrame({
    "Crop": train_crops,
    "Yield": y_train
})

test_summary = pd.DataFrame({
    "Crop": test_crops,
    "Yield": y_test
})

print("\nTRAINING VS TESTING YIELD DISTRIBUTION")

for crop in weak_crops:

    train_values = train_summary[
        train_summary["Crop"] == crop
    ]["Yield"]

    test_values = test_summary[
        test_summary["Crop"] == crop
    ]["Yield"]

    print("\nCrop:", crop)

    print("\nTraining statistics:")
    print(train_values.describe())

    print("\nTesting statistics:")
    print(test_values.describe())


# ==========================================
# 6. SAVE COMBINED SUMMARY
# ==========================================

summary = []

for crop in weak_crops:

    train_values = train_summary[
        train_summary["Crop"] == crop
    ]["Yield"]

    test_values = test_summary[
        test_summary["Crop"] == crop
    ]["Yield"]

    summary.append({
        "Crop": crop,
        "Train_Count": len(train_values),
        "Test_Count": len(test_values),
        "Train_Mean": train_values.mean(),
        "Test_Mean": test_values.mean(),
        "Train_Median": train_values.median(),
        "Test_Median": test_values.median(),
        "Train_Max": train_values.max(),
        "Test_Max": test_values.max()
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    DIAGNOSTICS_DIR / "train_test_crop_comparison.csv",
    index=False
)

print("\nERROR DIAGNOSTICS COMPLETED")
