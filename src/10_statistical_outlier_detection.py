
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "dataset" / "crop_yield_dataset_with_provenance.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "outlier_analysis"

# ==========================================
# 1. LOAD ORIGINAL DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

selected_crops = [
    "Rice",
    "Wheat",
    "Maize",
    "Sugarcane",
    "Cotton",
    "Groundnut",
    "Soybean",
    "Chickpea",
    "Pigeonpea",
    "Rapeseed & Mustard"
]

df = df[df["Crop"].isin(selected_crops)].copy()

print("Dataset loaded:", df.shape)


# ==========================================
# 2. CREATE OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. DETECT OUTLIERS USING IQR
# ==========================================

outlier_records = []
summary_records = []

for crop in selected_crops:

    crop_data = df[df["Crop"] == crop].copy()

    Q1 = crop_data["Yield"].quantile(0.25)
    Q3 = crop_data["Yield"].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    crop_data["Lower_Limit"] = lower_limit
    crop_data["Upper_Limit"] = upper_limit

    crop_data["Outlier_Status"] = np.where(
        (crop_data["Yield"] < lower_limit) |
        (crop_data["Yield"] > upper_limit),
        "Flagged",
        "Normal"
    )

    flagged = crop_data[
        crop_data["Outlier_Status"] == "Flagged"
    ]

    outlier_records.append(flagged)

    summary_records.append({
        "Crop": crop,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Limit": lower_limit,
        "Upper_Limit": upper_limit,
        "Total_Records": len(crop_data),
        "Flagged_Records": len(flagged),
        "Flagged_Percentage": (
            len(flagged) / len(crop_data) * 100
        )
    })


# ==========================================
# 4. SAVE FLAGGED RECORDS
# ==========================================

outlier_df = pd.concat(
    outlier_records,
    ignore_index=True
)

outlier_df.to_csv(
    OUTPUT_DIR / "flagged_records.csv",
    index=False
)


# ==========================================
# 5. SAVE CROP-WISE SUMMARY
# ==========================================

summary_df = pd.DataFrame(summary_records)

summary_df.to_csv(
    OUTPUT_DIR / "outlier_summary.csv",
    index=False
)

print("\nCROP-WISE OUTLIER SUMMARY")
print(summary_df.to_string(index=False))


# ==========================================
# 6. CHECK OUR KNOWN SUSPICIOUS RECORDS
# ==========================================

print("\nKNOWN SUSPICIOUS RECORDS")

known_records = df[
    (
        (df["Crop"] == "Chickpea") &
        (df["Year"] == 2007) &
        (df["Yield"] > 10000)
    ) |
    (
        (df["Crop"] == "Pigeonpea") &
        (df["Year"] == 2009) &
        (df["District"] == "Dhanbad")
    )
]

print(
    known_records[
        ["Year", "State", "District", "Crop", "Yield"]
    ].to_string(index=False)
)

print("\nSTATISTICAL OUTLIER DETECTION COMPLETED")
print("No observations have been deleted.")
