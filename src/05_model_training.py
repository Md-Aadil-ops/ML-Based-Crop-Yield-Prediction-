
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. LOAD TRAINING AND TESTING DATA
# ==========================================

X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv")
X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv")

y_train = pd.read_csv(OUTPUT_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(OUTPUT_DIR / "y_test.csv").squeeze()

print("DATA LOADED SUCCESSFULLY")
print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)


# ==========================================
# 2. DEFINE THREE ML ALGORITHMS
# ==========================================

models = {

    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree Regressor": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# ==========================================
# 3. CREATE OUTPUT DIRECTORIES
# ==========================================

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

results = []
predictions = pd.DataFrame({
    "Actual_Yield": y_test
})


# ==========================================
# 4. TRAIN AND EVALUATE MODELS
# ==========================================

for name, model in models.items():

    print("\n================================")
    print("TRAINING:", name)
    print("================================")

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    # Store predictions
    predictions[name] = y_pred

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        MODEL_DIR / filename
    )

    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    print("Model saved:", filename)


# ==========================================
# 5. SAVE EVALUATION RESULTS
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "model_evaluation_results.csv",
    index=False
)

predictions.to_csv(
    OUTPUT_DIR / "model_predictions.csv",
    index=False
)

print("\n================================")
print("MODEL TRAINING COMPLETED")
print("================================")

print("\nMODEL EVALUATION RESULTS:")
print(results_df.to_string(index=False))

print("\nAll three models trained and saved successfully!")
