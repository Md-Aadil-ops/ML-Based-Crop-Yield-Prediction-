
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load original dataset
input_path = PROJECT_DIR / "dataset" / "crop_yield_dataset_with_provenance.csv"

df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Define the three suspicious records
suspicious_records = [
    ("Chickpea", 2007, "Muzaffarpur"),
    ("Chickpea", 2007, "Saharsa"),
    ("Pigeonpea", 2009, "Dhanbad")
]

# Create a mask for records to remove
remove_mask = pd.Series(False, index=df.index)

for crop, year, district in suspicious_records:
    remove_mask |= (
        (df["Crop"] == crop) &
        (df["Year"] == year) &
        (df["District"] == district)
    )

# Display records being removed
removed_records = df[remove_mask].copy()

print("\nRECORDS REMOVED:")
print(
    removed_records[
        ["Year", "State", "District", "Crop", "Yield"]
    ].to_string(index=False)
)

# Create cleaned dataset
cleaned_df = df[~remove_mask].copy()

# Save separately
output_path = PROJECT_DIR / "dataset" / "crop_yield_dataset_cleaned.csv"

cleaned_df.to_csv(output_path, index=False)

# Display summary
print("\nCLEANING SUMMARY")
print("Original records:", len(df))
print("Removed records:", len(removed_records))
print("Remaining records:", len(cleaned_df))
print("Cleaned dataset shape:", cleaned_df.shape)

print("\nCleaned dataset saved at:", output_path)
print("\nOriginal dataset remains unchanged.")
