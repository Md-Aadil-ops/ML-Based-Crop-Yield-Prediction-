# Results Traceability and Empirical Audit

This document provides a complete audit trail connecting every numerical result, statistical table, figure, and claim in the research paper directly to the source code, output CSV files, and visual artifacts within the project directory.

---

## 1. Traceability of Numerical Data and Metrics

### 1.1 Dataset Composition and Cleaning
| Paper Claim / Number | Source File | Exact Script / Function | Source Column / Expression | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **198,369** original observations | `dataset/crop_yield_dataset_with_provenance.csv` | `src/01_dataset_inspection.py` | `df.shape[0]` | Verified |
| **20** original columns | `dataset/crop_yield_dataset_with_provenance.csv` | `src/01_dataset_inspection.py` | `len(df.columns)` | Verified |
| **21** original crops | `outputs/crop_distribution.csv` | `src/01_dataset_inspection.py` | `df['Crop'].value_counts()` | Verified |
| **1966–2017** time span (52 years) | `dataset/crop_yield_dataset_with_provenance.csv` | `src/01_dataset_inspection.py` | `df['Year'].min(), max()` | Verified |
| **3** removed suspicious records | `outputs/outlier_analysis/suspicious_record_details.csv` | `src/11_suspicious_record_validation.py`, `src/12_create_cleaned_dataset.py` | Rows: Chickpea (Muzaffarpur 2007: 58363.64), Chickpea (Saharsa 2007: 32000.0), Pigeonpea (Dhanbad 2009: 12402.6) | Verified |
| **198,366** cleaned records | `dataset/crop_yield_dataset_cleaned.csv` | `src/12_create_cleaned_dataset.py` | `len(cleaned_df)` | Verified |
| **115,338** raw records for 10 crops | Calculated from original dataset | `src/03_data_preprocessing.py` | Sum of counts for 10 crops | Verified |
| **115,335** processed 10-crop records | `outputs/processed_crop_yield.csv` | `src/03_data_preprocessing.py` | `len(df)` | Verified |
| **29** input feature columns | `outputs/X_train.csv` | `src/03_data_preprocessing.py`, `app.py` | `len(X_train.columns)` | Verified |
| **92,913** training records (80.56%) | `outputs/X_train.csv` | `src/04_train_test_split.py` | `len(X_train)` (Year <= 2007) | Verified |
| **22,422** testing records (19.44%) | `outputs/X_test.csv` | `src/04_train_test_split.py` | `len(X_test)` (Year >= 2008) | Verified |

---

### 1.2 Global Model Performance Evaluation
*Source File: `outputs/model_evaluation_results.csv`*  
*Generating Script: `src/05_model_training.py`*

| Algorithm | Stated Metric | Source CSV Column | Exact Numerical Value in CSV |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | MAE | `MAE` | `769.6851965137774` |
| | MSE | `MSE` | `1421233.6738296303` |
| | RMSE | `RMSE` | `1192.1550544411705` |
| | $R^2$ Score | `R2 Score` | `0.6358757305287552` |
| **Decision Tree Regressor** | MAE | `MAE` | `695.9376750512889` |
| | MSE | `MSE` | `1561169.108313072` |
| | RMSE | `RMSE` | `1249.4675299154724` |
| | $R^2$ Score | `R2 Score` | `0.600023858459664` |
| **Random Forest Regressor** | MAE | `MAE` | `548.5933832352154` |
| | MSE | `MSE` | `901183.2177796786` |
| | RMSE | `RMSE` | `949.3067037473604` |
| | $R^2$ Score | `R2 Score` | `0.7691141950291933` |

---

### 1.3 Crop-Wise Random Forest Performance
*Source File: `outputs/crop_wise_model_performance.csv`*  
*Generating Script: `src/06_prediction_analysis.py`*

| Crop | Records | MAE (kg/ha) | MSE ((kg/ha)²) | RMSE (kg/ha) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Chickpea** | 2551 | 253.70397420619364 | 124180.04206619445 | 352.3918870606905 | 0.18216753615422943 |
| **Cotton** | 1403 | 164.4617317177477 | 87380.868416946 | 295.6025514384915 | 0.051442680943234786 |
| **Groundnut** | 2249 | 392.7750505558026 | 286137.35717563704 | 534.9180845471922 | 0.2918852077647096 |
| **Maize** | 2681 | 859.8863886609473 | 1584569.3264079986 | 1258.7967772472246 | 0.39051061213702354 |
| **Pigeonpea** | 2460 | 277.05334365853656 | 151792.8576650481 | 389.6060287842683 | 0.13653562084259419 |
| **Rapeseed & Mustard** | 2267 | 256.44537234230256 | 125867.79034703784 | 354.7785088573402 | 0.4326940531311162 |
| **Rice** | 2703 | 517.489565445801 | 481242.2127798471 | 693.7162336141826 | 0.4853410951514735 |
| **Soybean** | 1225 | 352.69616759183674 | 202788.93212466987 | 450.32092126023844 | 0.06869033319389439 |
| **Sugarcane** | 2370 | 1582.016430464135 | 4805780.741833131 | 2192.2091008462517 | 0.339367038704138 |
| **Wheat** | 2513 | 553.4408153601273 | 508368.87037578353 | 712.9999090994216 | 0.574174590785741 |
| **Total / Global** | **22422** | **548.5933832352154** | **901183.2177796786** | **949.3067037473604** | **0.7691141950291933** |

---

### 1.4 Temporal Concept Drift Metrics (Weak Crops)
*Source File: `outputs/error_diagnostics/train_test_crop_comparison.csv`*  
*Generating Script: `src/08_error_diagnostics.py`*

| Crop | Train Count | Test Count | Train Mean | Test Mean | Train Max | Test Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chickpea** | 10920 | 2551 | 673.829596 | 959.984430 | 4833.33 | 5000.00 |
| **Pigeonpea** | 10207 | 2460 | 743.483649 | 850.698341 | 5608.70 | 3827.59 |
| **Cotton** | 6197 | 1403 | 228.323229 | 388.454768 | 5000.00 | 2857.74 |
| **Soybean** | 2798 | 1225 | 898.911355 | 1086.686318 | 3333.33 | 3884.06 |

---

### 1.5 Feature Importance Metrics
*Source File: `outputs/feature_importance/random_forest_feature_importance.csv`*  
*Generating Script: `src/14_feature_importance.py`*

* Top 1: `Crop_Sugarcane` = `0.2926913890783672` (29.27%)
* Top 2: `Season_Whole Year` = `0.29232602701360116` (29.23%)
* Top 3: `Humidity` = `0.07213463024655112` (7.21%)
* Top 4: `Potassium` = `0.04266521906043337` (4.27%)
* Top 5: `Fertilizer` = `0.03555786423387923` (3.56%)
* Top 6: `Temperature` = `0.03520639925623997` (3.52%)
* Top 7: `Rainfall` = `0.031294794851090504` (3.13%)
* Top 8: `Nitrogen` = `0.030708677741968397` (3.07%)
* Top 9: `pH` = `0.03068882754137854` (3.07%)
* Top 10: `Year` = `0.021656274944733843` (2.17%)

---

## 2. Traceability of Figures in Research Paper

| Paper Figure | Source File in Project | Generating Script | Description |
| :--- | :--- | :--- | :--- |
| **Figure 1** | `outputs/01_crop_distribution.png` | `src/02_EDA.py` | Bar chart showing distribution of records across all 21 crops in the raw dataset. |
| **Figure 2** | `outputs/09_correlation_heatmap.png` | `src/02_EDA.py` | Heatmap of Pearson correlation coefficients between numerical climatic and soil features. |
| **Figure 3** | `outputs/10_actual_vs_predicted.png` | `src/06_prediction_analysis.py` | Scatterplot of actual vs. predicted yield for Random Forest on 2008–2017 test set. |
| **Figure 4** | `outputs/11_prediction_error_distribution.png` | `src/06_prediction_analysis.py` | Histogram and KDE curve of residuals ($Actual - Predicted$) in kg/ha. |
| **Figure 5** | `outputs/13_crop_wise_mae.png` | `src/06_prediction_analysis.py` | Bar plot illustrating Mean Absolute Error across all 10 evaluated crops. |
| **Figure 6** | `outputs/feature_importance/random_forest_feature_importance.png` | `src/14_feature_importance.py` | Horizontal bar chart of Gini feature importance across all 29 features. |
| **Figure 7** | `outputs/weak_crop_analysis/cotton_actual_vs_predicted.png` | `src/07_weak_crop_analysis.py` | Scatterplot of Cotton actual vs. predicted yield showing the extrapolation ceiling. |
| **Figure 8** | `outputs/weak_crop_analysis/chickpea_actual_vs_predicted.png` | `src/07_weak_crop_analysis.py` | Scatterplot of Chickpea actual vs. predicted yield showing wide residual spread. |

---

## 3. Verification Confirmation

All data, tables, figures, record counts, and metrics cited in the research paper have been 100% verified against original codebase files. Zero values have been approximated, invented, or modified.
