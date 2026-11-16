# Crop Yield Prediction

This project is a complete crop yield prediction system built in Python for agricultural data analysis and forecasting. It walks through the full machine learning workflow: dataset inspection, exploratory data analysis, preprocessing, model training, evaluation, advanced diagnostics, and a Streamlit-based prediction app.

## Overview

The goal is to predict crop yield (kg/ha) using environmental and agricultural features such as:

- crop type
- season
- soil type
- temperature
- rainfall
- humidity
- soil pH
- nitrogen, phosphorus, potassium
- fertilizer use
- pesticide use
- year

The project evaluates multiple regression models and compares their performance to identify the most reliable predictor.

## What has already been implemented

### 1. Dataset inspection and understanding
- Original dataset and cleaned dataset are stored under `dataset/`
- Exploratory validation and inspection scripts are included in `src/01_dataset_inspection.py`
- Dataset structure, columns, and quality checks were performed before modeling

### 2. Exploratory data analysis (EDA)
- `src/02_EDA.py` analyzes distributions, relationships, seasonal trends, and crop-wise patterns
- Visual outputs were generated in `outputs/` such as:
  - `01_crop_distribution.png`
  - `02_yield_distribution.png`
  - `03_yield_boxplot.png`
  - `04_state_yield.png`
  - `05_yearly_yield.png`
  - `06_rainfall_yield.png`
  - `07_temperature_yield.png`
  - `08_soil_yield.png`
  - `09_correlation_heatmap.png`

### 3. Data cleaning and preprocessing
- `src/03_data_preprocessing.py` handles data cleaning and feature preparation
- `src/12_create_cleaned_dataset.py` creates a cleaned and structured dataset for model training
- The processed output is saved as:
  - `outputs/processed_crop_yield.csv`

### 4. Train/test split and feature preparation
- `src/04_train_test_split.py` creates train and test sets
- Generated datasets:
  - `outputs/X_train.csv`
  - `outputs/X_test.csv`
  - `outputs/y_train.csv`
  - `outputs/y_test.csv`

### 5. Machine learning model training
- `src/05_model_training.py` trains and evaluates three regression algorithms:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Trained models are saved in `models/`:
  - `linear_regression.pkl`
  - `decision_tree_regressor.pkl`
  - `random_forest_regressor.pkl`

### 6. Prediction analysis and performance evaluation
- `src/06_prediction_analysis.py` evaluates model predictions and produces key analytics
- Outputs include:
  - `outputs/model_evaluation_results.csv`
  - `outputs/model_predictions.csv`
  - `outputs/random_forest_prediction_analysis.csv`
  - `outputs/crop_wise_model_performance.csv`
  - `outputs/10_actual_vs_predicted.png`
  - `outputs/11_prediction_error_distribution.png`
  - `outputs/12_residual_analysis.png`
  - `outputs/13_crop_wise_mae.png`

### 7. Weak crop analysis and diagnostics
- `src/07_weak_crop_analysis.py` investigates crops with weaker predictive accuracy
- `src/08_error_diagnostics.py` explores error patterns and model weaknesses
- Output visuals for weak-crop investigation are saved under `outputs/weak_crop_analysis/`
- Error diagnostic CSVs are saved under `outputs/error_diagnostics/`

### 8. Outlier and suspicious-record investigation
- `src/09_outlier_investigation.py`
- `src/10_statistical_outlier_detection.py`
- `src/11_suspicious_record_validation.py`
- Outputs saved under:
  - `outputs/outlier_analysis/`
  - `outputs/flagged_records.csv`
  - `outputs/suspicious_record_details.csv`

### 9. Feature importance analysis
- `src/14_feature_importance.py` measures the influence of features on the random forest prediction model
- Results are stored in:
  - `outputs/feature_importance/random_forest_feature_importance.csv`
  - `outputs/feature_importance/random_forest_feature_importance.png`

### 10. Interactive prediction app
- `app.py` provides a Streamlit web interface for yield prediction
- Users can select crop, season, soil type, and agricultural/environmental values
- The app loads one of the trained models and generates a predicted yield in kg/ha

## Current model performance

The evaluation results already generated in `outputs/model_evaluation_results.csv` show the following:

| Model | MAE | RMSE | R² Score |
| --- | ---: | ---: | ---: |
| Linear Regression | 769.69 | 1192.16 | 0.6359 |
| Decision Tree Regressor | 695.94 | 1249.47 | 0.6000 |
| Random Forest Regressor | 548.59 | 949.31 | 0.7691 |

The Random Forest model currently provides the best predictive performance in this project.

## Project structure

- `dataset/` — original and cleaned crop yield datasets
- `src/` — all Python scripts for preprocessing, analysis, feature engineering, and modeling
- `models/` — trained machine learning models
- `outputs/` — generated datasets, plots, evaluation reports, and diagnostics
- `app.py` — Streamlit interface for making crop yield predictions
- `requirements.txt` — required Python dependencies

## Dependencies

The project uses the following core libraries:

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- streamlit

## Setup

1. Open the project folder:
   ```bash
   cd "C:\Users\MD AADIL\Downloads\Predictive analysis Project\Crop_Yield_Prediction"
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the application

From the project directory, launch the app:

```bash
streamlit run app.py
```

This will open the crop yield prediction dashboard where you can enter input values and generate predictions.

## Summary

This repository already contains a full machine learning pipeline for crop yield prediction, including:

- data cleaning and preprocessing
- exploratory analysis and visual diagnostics
- train/test preparation
- model training and evaluation
- advanced outlier and error analysis
- feature importance analysis
- a ready-to-use Streamlit prediction app

It is structured and ready for further improvements such as hyperparameter tuning, model comparison studies, deployment, or integration with more agricultural data sources.
