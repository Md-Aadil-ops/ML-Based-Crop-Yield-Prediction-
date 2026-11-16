
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "dataset" / "crop_yield_dataset_with_provenance.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "outlier_analysis"

# Load original dataset
df = pd.read_csv(DATASET_PATH)

# Suspicious records identified during IQR analysis
suspicious = df[
    (
        (df["Crop"] == "Chickpea") &
        (df["Year"] == 2007) &
        (df["District"].isin(["Muzaffarpur", "Saharsa"]))
    )
    |
    (
        (df["Crop"] == "Pigeonpea") &
        (df["Year"] == 2009) &
        (df["District"] == "Dhanbad")
    )
]

# Display complete original records
print("\nSUSPICIOUS RECORD DETAILS")
print(suspicious.to_string(index=False))

# Check other crops in the same district and year
print("\nOTHER CROPS IN THE SAME DISTRICT-YEARS")

for _, row in suspicious.iterrows():

    related = df[
        (df["District"] == row["District"]) &
        (df["Year"] == row["Year"])
    ]

    print(
        f"\nDistrict: {row['District']} | "
        f"Year: {row['Year']}"
    )

    print(
        related[
            ["Crop", "Yield", "Rainfall",
             "Temperature", "Fertilizer", "Pesticide"]
        ].to_string(index=False)
    )

# Save validation records
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
suspicious.to_csv(OUTPUT_DIR / "suspicious_record_details.csv", index=False)

print("\nValidation investigation completed.")
