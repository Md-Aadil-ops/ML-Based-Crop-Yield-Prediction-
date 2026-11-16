
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD TEST DATA AND PREDICTIONS
# ==========================================

X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv")

predictions = pd.read_csv(
    OUTPUT_DIR / "model_predictions.csv"
)

y_actual = predictions["Actual_Yield"]
y_pred = predictions["Random Forest Regressor"]

# Reconstruct crop names from one-hot encoded columns
crop_columns = [
    col for col in X_test.columns
    if col.startswith("Crop_")
]

if not crop_columns:
    raise ValueError("Crop one-hot columns not found in X_test.csv")

crop_names = (
    X_test[crop_columns]
    .idxmax(axis=1)
    .str.replace("Crop_", "", regex=False)
)

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# 2. CREATE ANALYSIS DATAFRAME
# ==========================================

analysis_df = pd.DataFrame({
    "Crop": crop_names,
    "Actual_Yield": y_actual,
    "Predicted_Yield": y_pred
})

analysis_df["Residual"] = (
    analysis_df["Actual_Yield"]
    - analysis_df["Predicted_Yield"]
)

analysis_df["Absolute_Error"] = (
    analysis_df["Residual"].abs()
)

analysis_df.to_csv(
    OUTPUT_DIR / "random_forest_prediction_analysis.csv",
    index=False
)

# ==========================================
# 3. ACTUAL VS PREDICTED YIELD
# ==========================================

plt.figure(figsize=(9, 7))

sns.scatterplot(
    data=analysis_df.sample(
        min(5000, len(analysis_df)),
        random_state=42
    ),
    x="Actual_Yield",
    y="Predicted_Yield",
    alpha=0.5
)

# Perfect prediction reference line
minimum = min(y_actual.min(), y_pred.min())
maximum = max(y_actual.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted Crop Yield - Random Forest")
plt.xlabel("Actual Yield (kg/ha)")
plt.ylabel("Predicted Yield (kg/ha)")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_actual_vs_predicted.png")
plt.show()


# ==========================================
# 4. PREDICTION ERROR DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    analysis_df["Residual"],
    bins=50,
    kde=True
)

plt.axvline(
    0,
    linestyle="--",
    label="Zero Error"
)

plt.title("Distribution of Prediction Errors")
plt.xlabel("Residual (Actual - Predicted) in kg/ha")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_prediction_error_distribution.png")
plt.show()


# ==========================================
# 5. RESIDUAL VS PREDICTED YIELD
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=analysis_df.sample(
        min(5000, len(analysis_df)),
        random_state=42
    ),
    x="Predicted_Yield",
    y="Residual",
    alpha=0.5
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title("Residual Analysis - Random Forest")
plt.xlabel("Predicted Yield (kg/ha)")
plt.ylabel("Residual (kg/ha)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "12_residual_analysis.png")
plt.show()


# ==========================================
# 6. CROP-WISE PREDICTION PERFORMANCE
# ==========================================

crop_results = []

for crop, group in analysis_df.groupby("Crop"):

    actual = group["Actual_Yield"]
    predicted = group["Predicted_Yield"]

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    r2 = r2_score(actual, predicted)

    crop_results.append({
        "Crop": crop,
        "Records": len(group),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

crop_results_df = pd.DataFrame(crop_results)

crop_results_df = crop_results_df.sort_values("Crop")

print("\nCROP-WISE MODEL PERFORMANCE")
print(crop_results_df.to_string(index=False))

crop_results_df.to_csv(
    OUTPUT_DIR / "crop_wise_model_performance.csv",
    index=False
)


# ==========================================
# 7. CROP-WISE MAE VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 7))

sns.barplot(
    data=crop_results_df,
    x="MAE",
    y="Crop"
)

plt.title("Random Forest - MAE by Crop")
plt.xlabel("Mean Absolute Error (kg/ha)")
plt.ylabel("Crop")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "13_crop_wise_mae.png")
plt.show()


# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\nOVERALL RANDOM FOREST PERFORMANCE")

print(
    "MAE:",
    round(mean_absolute_error(y_actual, y_pred), 2)
)

print(
    "RMSE:",
    round(np.sqrt(mean_squared_error(y_actual, y_pred)), 2)
)

print(
    "R2 Score:",
    round(r2_score(y_actual, y_pred), 4)
)

print("\nPREDICTION ANALYSIS COMPLETED")
print("All analysis files saved in outputs/")
