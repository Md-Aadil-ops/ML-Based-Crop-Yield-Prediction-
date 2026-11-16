
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "dataset" / "crop_yield_dataset_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

print("Original Dataset Shape:", df.shape)


# ==========================================
# 2. SELECT 10 MAJOR CROPS
# ==========================================

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

print("\nDataset Shape After Crop Selection:", df.shape)


# ==========================================
# 3. REMOVE UNNECESSARY COLUMNS
# ==========================================

columns_to_remove = [
    "Region",
    "State",
    "District",
    "temp_src",
    "rain_src",
    "pest_src"
]

df.drop(
    columns=columns_to_remove,
    inplace=True
)

print("\nRemaining Columns:")
print(df.columns.tolist())


# ==========================================
# 4. CHECK MISSING VALUES
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# 5. CHECK DUPLICATES
# ==========================================

print("\nDuplicate Records:", df.duplicated().sum())


# ==========================================
# 6. ENCODE CATEGORICAL VARIABLES
# ==========================================

df = pd.get_dummies(
    df,
    columns=["Crop", "Season", "Soil Type"],
    dtype=int
)

print("\nDataset After Encoding:", df.shape)


# ==========================================
# 7. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Yield"])
y = df["Yield"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# ==========================================
# 8. SAVE PROCESSED DATA
# ==========================================

OUTPUT_DIR.mkdir(exist_ok=True)

df.to_csv(
    OUTPUT_DIR / "processed_crop_yield.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")

print("\nDATA PREPROCESSING COMPLETED")
