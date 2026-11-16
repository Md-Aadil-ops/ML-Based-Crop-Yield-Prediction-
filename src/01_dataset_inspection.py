import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------
# 1. Load dataset
# --------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "dataset"

provenance_file = DATA_DIR / "crop_yield_dataset_with_provenance.csv"
main_file = DATA_DIR / "crop_yield_dataset.csv"

if provenance_file.exists():
    file_path = provenance_file
elif main_file.exists():
    file_path = main_file
else:
    raise FileNotFoundError(
        "Dataset CSV not found. Place your CSV inside the dataset folder."
    )

df = pd.read_csv(file_path)

print("\nDATASET LOADED SUCCESSFULLY")
print("File:", file_path.name)

# --------------------------------
# 2. Basic dataset information
# --------------------------------

print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. FIRST FIVE RECORDS")
print(df.head())

print("\n3. LAST FIVE RECORDS")
print(df.tail())

print("\n4. COLUMN NAMES")
print(df.columns.tolist())

print("\n5. DATASET INFORMATION")
df.info()

print("\n6. STATISTICAL SUMMARY")
print(df.describe(include="all").T)

# --------------------------------
# 3. Missing values
# --------------------------------

print("\n7. MISSING VALUES")
print(df.isnull().sum())

print("\nMissing value percentage:")
print((df.isnull().mean() * 100).round(2))

# --------------------------------
# 4. Duplicate records
# --------------------------------

print("\n8. DUPLICATE RECORDS")
print(df.duplicated().sum())

# --------------------------------
# 5. Unique values
# --------------------------------

print("\n9. UNIQUE VALUES")

for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

# --------------------------------
# 6. Crop analysis
# --------------------------------

print("\n10. CROP DISTRIBUTION")

crop_counts = df["Crop"].value_counts()

print(crop_counts)

crop_counts.to_csv(
    PROJECT_DIR / "outputs" / "crop_distribution.csv"
) if (PROJECT_DIR / "outputs").exists() else None

# --------------------------------
# 7. State analysis
# --------------------------------

print("\n11. STATE DISTRIBUTION")

if "State" in df.columns:
    print(df["State"].value_counts())

# --------------------------------
# 8. Yield analysis
# --------------------------------

print("\n12. YIELD STATISTICS")

print(df["Yield"].describe())

print("\nZero yield records:", (df["Yield"] == 0).sum())
print("Negative yield records:", (df["Yield"] < 0).sum())

# --------------------------------
# 9. Dataset time period
# --------------------------------

print("\n13. YEAR RANGE")

print("Minimum year:", df["Year"].min())
print("Maximum year:", df["Year"].max())

# --------------------------------
# 10. Save inspection summary
# --------------------------------

print("\nDATASET INSPECTION COMPLETED")
