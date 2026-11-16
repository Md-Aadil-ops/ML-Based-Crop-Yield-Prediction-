
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "dataset" / "crop_yield_dataset_with_provenance.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

print("DATASET LOADED SUCCESSFULLY")
print("Dataset Shape:", df.shape)


# ==========================================
# 1. CROP-WISE RECORD DISTRIBUTION
# ==========================================

plt.figure(figsize=(13, 7))

crop_counts = df["Crop"].value_counts()

sns.barplot(
    x=crop_counts.values,
    y=crop_counts.index
)

plt.title("Crop-wise Record Distribution")
plt.xlabel("Number of Records")
plt.ylabel("Crop")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_crop_distribution.png")
plt.show()


# ==========================================
# 2. YIELD DISTRIBUTION HISTOGRAM
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Yield"],
    bins=50,
    kde=True
)

plt.title("Distribution of Crop Yield")
plt.xlabel("Yield (kg/ha)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_yield_distribution.png")
plt.show()


# ==========================================
# 3. YIELD BOXPLOT AND OUTLIER INVESTIGATION
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["Yield"]
)

plt.title("Crop Yield - Outlier Investigation")
plt.xlabel("Yield (kg/ha)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_yield_boxplot.png")
plt.show()


# ==========================================
# 4. STATE-WISE AVERAGE CROP YIELD
# ==========================================

state_yield = (
    df.groupby("State")["Yield"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 7))

sns.barplot(
    x=state_yield.values,
    y=state_yield.index
)

plt.title("State-wise Average Crop Yield")
plt.xlabel("Average Yield (kg/ha)")
plt.ylabel("State")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_state_yield.png")
plt.show()


# ==========================================
# 5. YEAR-WISE YIELD TRENDS
# ==========================================

yearly_yield = (
    df.groupby("Year")["Yield"]
    .mean()
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    x=yearly_yield.index,
    y=yearly_yield.values
)

plt.title("Year-wise Average Crop Yield Trend")
plt.xlabel("Year")
plt.ylabel("Average Yield (kg/ha)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_yearly_yield.png")
plt.show()


# ==========================================
# 6. RAINFALL VS. CROP YIELD
# ==========================================

sample = df.sample(
    n=min(5000, len(df)),
    random_state=42
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample,
    x="Rainfall",
    y="Yield",
    alpha=0.4
)

plt.title("Rainfall vs Crop Yield")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Yield (kg/ha)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_rainfall_yield.png")
plt.show()


# ==========================================
# 7. TEMPERATURE VS. CROP YIELD
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=sample,
    x="Temperature",
    y="Yield",
    alpha=0.4
)

plt.title("Temperature vs Crop Yield")
plt.xlabel("Temperature (°C)")
plt.ylabel("Yield (kg/ha)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_temperature_yield.png")
plt.show()


# ==========================================
# 8. SOIL TYPE VS. CROP YIELD
# ==========================================

plt.figure(figsize=(11, 6))

sns.boxplot(
    data=df,
    x="Soil Type",
    y="Yield"
)

plt.title("Soil Type vs Crop Yield")
plt.xlabel("Soil Type")
plt.ylabel("Yield (kg/ha)")

plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_soil_yield.png")
plt.show()


# ==========================================
# 9. CORRELATION HEATMAP
# ==========================================

numerical_df = df.select_dtypes(
    include="number"
)

plt.figure(figsize=(12, 9))

sns.heatmap(
    numerical_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_correlation_heatmap.png")
plt.show()


# ==========================================
# EDA COMPLETION
# ==========================================

print("\n================================")
print("EDA COMPLETED SUCCESSFULLY")
print("Total EDA Tasks Completed: 9")
print("Charts saved in outputs folder")
print("================================")
