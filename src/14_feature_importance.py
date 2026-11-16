
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# ==========================================
# 1. LOAD TRAINING FEATURES AND MODEL
# ==========================================

X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv")

model = joblib.load(
    PROJECT_DIR / "models" / "random_forest_regressor.pkl"
)

# ==========================================
# 2. EXTRACT FEATURE IMPORTANCE
# ==========================================

# Access Random Forest inside the model
# if it is stored directly or inside a Pipeline.

if hasattr(model, "feature_importances_"):
    importance_values = model.feature_importances_

elif hasattr(model, "named_steps"):
    importance_values = model.named_steps["model"].feature_importances_

else:
    raise ValueError("Random Forest feature importance not found.")

# Check feature count
if len(importance_values) != len(X_train.columns):
    raise ValueError(
        "Feature count mismatch between model and training data."
    )

# ==========================================
# 3. CREATE IMPORTANCE DATAFRAME
# ==========================================

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# ==========================================
# 4. DISPLAY TOP 15 FEATURES
# ==========================================

print("\nTOP 15 IMPORTANT FEATURES")
print(
    importance_df.head(15).to_string(index=False)
)

# ==========================================
# 5. SAVE FEATURE IMPORTANCE
# ==========================================

output_dir = OUTPUT_DIR / "feature_importance"
output_dir.mkdir(parents=True, exist_ok=True)

importance_df.to_csv(
    output_dir / "random_forest_feature_importance.csv",
    index=False
)

# ==========================================
# 6. VISUALIZE TOP 15 FEATURES
# ==========================================

top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(11, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 15 Features - Random Forest Model")
plt.tight_layout()

plt.savefig(
    output_dir / "random_forest_feature_importance.png",
    dpi=300
)

plt.show()

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED")
print("Results saved in:", output_dir)
