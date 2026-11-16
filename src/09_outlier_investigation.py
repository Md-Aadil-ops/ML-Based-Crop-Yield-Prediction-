
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "dataset" / "crop_yield_dataset_with_provenance.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# ==========================================
# 1. LOAD ORIGINAL DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

# ==========================================
# 2. INVESTIGATE EXTREME OBSERVATIONS
# ==========================================

print("\nCHICKPEA - EXTREME TRAINING YIELD")

chickpea = df[
    (df["Crop"] == "Chickpea") &
    (df["Year"] <= 2007) &
    (df["Yield"] > 5000)
]

print(chickpea.to_string(index=False))


print("\nPIGEONPEA - EXTREME TESTING YIELD")

pigeonpea = df[
    (df["Crop"] == "Pigeonpea") &
    (df["Year"] >= 2008) &
    (df["Yield"] > 5000)
]

print(pigeonpea.to_string(index=False))


print("\nCOTTON - HIGH YIELD IN 2016-2017")

cotton = df[
    (df["Crop"] == "Cotton") &
    (df["Year"].isin([2016, 2017])) &
    (df["Yield"] > 2000)
]

print(cotton.to_string(index=False))


print("\nSOYBEAN - HIGH YIELD OBSERVATIONS")

soybean = df[
    (df["Crop"] == "Soybean") &
    (df["Yield"] > 2500)
]

print(soybean.to_string(index=False))


# ==========================================
# 3. SAVE INVESTIGATION RESULTS
# ==========================================

results = pd.concat(
    [chickpea, pigeonpea, cotton, soybean]
).drop_duplicates()

OUTPUT_DIR.mkdir(exist_ok=True)
results.to_csv(OUTPUT_DIR / "extreme_yield_investigation.csv", index=False)

print("\nINVESTIGATION COMPLETED")
print("Results saved successfully.")
