
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

PROJECT_SUBDIR = BASE_DIR / "Crop_Yield_Prediction"
if (
    not (BASE_DIR / "models").is_dir()
    and (PROJECT_SUBDIR / "models").is_dir()
    and (PROJECT_SUBDIR / "outputs").is_dir()
):
    BASE_DIR = PROJECT_SUBDIR

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_PATH = OUTPUT_DIR / "X_train.csv"
PROCESSED_PATH = OUTPUT_DIR / "processed_crop_yield.csv"

print("PROJECT:", BASE_DIR)
print("OUTPUT:", OUTPUT_DIR)
print("X_train exists:", TRAIN_PATH.exists())
print("Processed file exists:", PROCESSED_PATH.exists())
print("Files inside outputs:", list(OUTPUT_DIR.glob("*")))

# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# ==========================================
# 3. LOAD FEATURE INFORMATION
# ==========================================

@st.cache_data
def load_feature_data():
    X_train = pd.read_csv(TRAIN_PATH)
    processed = pd.read_csv(PROCESSED_PATH)

    return X_train, processed


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


try:
    X_train, processed = load_feature_data()

    feature_columns = X_train.columns.tolist()

    if len(feature_columns) != 29:
        st.error(
            f"Expected 29 features, found {len(feature_columns)}."
        )
        st.stop()

except Exception as e:
    st.error(f"Could not load training feature information: {e}")
    st.stop()

# Get categorical options from the actual processed dataset
crop_options = sorted(
    col.replace("Crop_", "")
    for col in feature_columns
    if col.startswith("Crop_")
)

season_options = sorted(
    col.replace("Season_", "")
    for col in feature_columns
    if col.startswith("Season_")
)

soil_options = sorted(
    col.replace("Soil Type_", "")
    for col in feature_columns
    if col.startswith("Soil Type_")
)

categorical_prefixes = ("Crop_", "Season_", "Soil Type_")

numeric_features = [
    col for col in feature_columns
    if not col.startswith(categorical_prefixes)
]

# ==========================================
# 4. APPLICATION HEADER
# ==========================================

st.title("🌾 Crop Yield Prediction System")

st.write(
    "Predict agricultural crop yield in India "
    "using machine learning algorithms."
)

st.divider()

# ==========================================
# 5. SIDEBAR
# ==========================================

st.sidebar.title("Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Machine Learning Algorithm",
    [
        "Random Forest Regressor",
        "Decision Tree Regressor",
        "Linear Regression"
    ]
)

model_files = {
    "Random Forest Regressor": "random_forest_regressor.pkl",
    "Decision Tree Regressor": "decision_tree_regressor.pkl",
    "Linear Regression": "linear_regression.pkl"
}

st.sidebar.info(
    "Choose one of the three trained models "
    "to generate a crop yield prediction."
)

# ==========================================
# 6. INPUT FORM
# ==========================================

st.subheader("Enter Agricultural Information")

with st.form("crop_prediction_form"):

    st.markdown("### Crop Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        crop = st.selectbox("Select Crop", crop_options)

    with col2:
        season = st.selectbox("Select Season", season_options)

    with col3:
        soil_type = st.selectbox("Select Soil Type", soil_options)

    st.markdown("### Environmental Conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input(
            "Temperature (°C)",
            value=25.0,
            step=0.5
        )

    with col2:
        rainfall = st.number_input(
            "Rainfall (mm)",
            value=1000.0,
            min_value=0.0
        )

    with col3:
        humidity = st.number_input(
            "Humidity (%)",
            value=60.0,
            min_value=0.0,
            max_value=100.0
        )

    st.markdown("### Soil and Agricultural Inputs")

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input(
            "Soil pH",
            value=6.5,
            min_value=0.0,
            max_value=14.0,
            step=0.1
        )

    with col2:
        nitrogen = st.number_input(
            "Nitrogen (kg/ha)",
            value=50.0,
            min_value=0.0
        )

    with col3:
        phosphorus = st.number_input(
            "Phosphorus (kg/ha)",
            value=20.0,
            min_value=0.0
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        potassium = st.number_input(
            "Potassium (kg/ha)",
            value=10.0,
            min_value=0.0
        )

    with col2:
        fertilizer = st.number_input(
            "Fertilizer (kg/ha)",
            value=100.0,
            min_value=0.0
        )

    with col3:
        pesticide = st.number_input(
            "Pesticide (kg/ha)",
            value=0.1,
            min_value=0.0
        )

    st.markdown("### Year")

    year = st.number_input(
    "Select Year",
    min_value=1966,
    max_value=2026,
    value=2017,
    step=1
)

    submitted = st.form_submit_button(
        "🌱 Predict Crop Yield",
        use_container_width=True
    )

# ==========================================
# 7. PREDICTION
# ==========================================

if submitted:

    try:
        # Numeric inputs
        input_values = {
            "Year": year,
            "Temperature": temperature,
            "Rainfall": rainfall,
            "Humidity": humidity,
            "pH": ph,
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "Fertilizer": fertilizer,
            "Pesticide": pesticide
        }

        # Create a row using the exact training feature order
        input_df = pd.DataFrame(
            0.0,
            index=[0],
            columns=feature_columns,
            dtype="float64"
        )

        # Fill numerical features
        for feature in numeric_features:
            if feature not in input_values:
                raise ValueError(
                    f"Missing input value for feature: {feature}"
                )

            input_df.loc[0, feature] = input_values[feature]

        # One-hot encode categorical inputs
        categorical_values = {
            f"Crop_{crop}": 1,
            f"Season_{season}": 1,
            f"Soil Type_{soil_type}": 1
        }

        for feature, value in categorical_values.items():
            if feature in input_df.columns:
                input_df.loc[0, feature] = value
            else:
                raise ValueError(
                    f"Unknown category feature: {feature}"
                )

        # Load selected model
        selected_model_path = (
            MODEL_DIR / model_files[model_choice]
        )

        model = load_model(selected_model_path)

        # Generate prediction
        prediction = float(model.predict(input_df)[0])

        st.divider()
        st.subheader("Crop Yield Prediction")

        if prediction < 0:

            st.error(
                f"Invalid prediction: {prediction:,.2f} kg/ha"
            )

            st.warning(
                "The selected model generated a negative yield, "
                "which is not physically meaningful. "
                "Try Random Forest or Decision Tree."
            )

        else:

            st.success("Crop Yield Prediction Completed!")

            st.metric(
                label=f"{crop} Estimated Yield",
                value=f"{prediction:,.2f} kg/ha"
            )

            st.write(f"**Selected Crop:** {crop}")
            st.write(f"**Selected Season:** {season}")
            st.write(f"**Selected Soil Type:** {soil_type}")
            st.write(f"**Selected Algorithm:** {model_choice}")

            st.caption(
                "This is a machine-learning estimate, not a "
                "guaranteed agricultural outcome."
            )


        

        if prediction < 0:
            st.warning(
                "The model returned a negative yield. "
                "This is not a physically meaningful crop yield."
            )

        st.caption(
            "This is a machine-learning estimate, not a "
            "guaranteed agricultural outcome. Predictions "
            "depend on the quality and coverage of historical data."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "Crop Yield Prediction System | "
    "Developed using Python, Streamlit and Scikit-learn"
)
