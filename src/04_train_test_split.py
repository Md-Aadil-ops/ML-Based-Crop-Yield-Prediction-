
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# ==========================================
# 1. LOAD PROCESSED DATASET
# ==========================================

df = pd.read_csv(OUTPUT_DIR / "processed_crop_yield.csv")

print("DATASET LOADED SUCCESSFULLY")
print("Original Shape:", df.shape)


# ==========================================
# 2. CHECK YEAR RANGE
# ==========================================

print("\nYEAR RANGE")
print("Minimum Year:", df["Year"].min())
print("Maximum Year:", df["Year"].max())


# ==========================================
# 3. CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================

# Training: 1966 to 2007
# Testing: 2008 to 2017

train_df = df[df["Year"] <= 2007].copy()

test_df = df[df["Year"] >= 2008].copy()


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X_train = train_df.drop(columns=["Yield"])
y_train = train_df["Yield"]

X_test = test_df.drop(columns=["Yield"])
y_test = test_df["Yield"]


# ==========================================
# 5. DISPLAY DATASET INFORMATION
# ==========================================

print("\nTRAINING DATA")
print("X_train Shape:", X_train.shape)
print("y_train Shape:", y_train.shape)
print("Training Year Range:",
      X_train["Year"].min(), "-", X_train["Year"].max())


print("\nTESTING DATA")
print("X_test Shape:", X_test.shape)
print("y_test Shape:", y_test.shape)
print("Testing Year Range:",
      X_test["Year"].min(), "-", X_test["Year"].max())


# ==========================================
# 6. CHECK DATA PROPORTIONS
# ==========================================

total_records = len(df)

train_percentage = len(train_df) / total_records * 100
test_percentage = len(test_df) / total_records * 100

print("\nDATA SPLIT PERCENTAGE")
print(f"Training: {train_percentage:.2f}%")
print(f"Testing: {test_percentage:.2f}%")


# ==========================================
# 7. CHECK TARGET DISTRIBUTION
# ==========================================

print("\nTRAINING TARGET STATISTICS")
print(y_train.describe())

print("\nTESTING TARGET STATISTICS")
print(y_test.describe())


# ==========================================
# 8. SAVE TRAINING AND TESTING DATA
# ==========================================

OUTPUT_DIR.mkdir(exist_ok=True)

X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)


# ==========================================
# 9. FINAL VERIFICATION
# ==========================================

assert X_train["Year"].max() < X_test["Year"].min()
assert len(X_train) + len(X_test) == len(df)

print("\nTRAIN-TEST SPLITTING COMPLETED SUCCESSFULLY")
print("Training and testing files saved in outputs/")
