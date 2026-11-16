# Crop Yield Prediction Project: Dataset and Methodology Facts

This document compiles the verified empirical facts, dataset specifications, preprocessing steps, and experimental configurations extracted directly from the codebase and outputs of the project: **"Crop Yield Prediction in India Using Machine Learning Algorithms"**.

---

## 1. Dataset Origin and Metadata

* **Base Repository / Origin:** ICRISAT District-Level Database (DLD) for Indian Agriculture, integrated with NASA POWER agroclimatological indicators and FAO national statistics.
* **Temporal Coverage:** 1966 to 2017 (52 continuous agricultural years).
* **Geographical Scope:** Multi-state, district-level historical agricultural observations across India.
* **Original Dataset Dimensions:** 198,369 rows × 20 columns (`dataset/crop_yield_dataset_with_provenance.csv`).
* **Original Crop Categories (21 crops):** Rice (14,476), Maize (13,615), Chickpea (13,473), Minor Pulses (13,302), Sugarcane (13,264), Wheat (13,227), Sesamum (12,862), Pigeonpea (12,668), Groundnut (11,796), Rapeseed & Mustard (11,196), Sorghum (Kharif) (11,053), Pearl Millet (9,796), Barley (7,673), Cotton (7,600), Linseed (6,755), Finger Millet (5,615), Castor (4,891), Sunflower (4,846), Soybean (4,023), Sorghum (Rabi) (3,948), Safflower (2,290).

---

## 2. Crop Selection and Data Cleaning

### 2.1 Selected Target Crops (10 Crops)
To focus the modeling on major staple and commercial crops, 10 primary crops were selected:
1. **Rice** (Staple Kharif cereal)
2. **Wheat** (Staple Rabi cereal)
3. **Maize** (Coarse grain cereal)
4. **Sugarcane** (Perennial cash crop)
5. **Cotton** (Cash fibre crop)
6. **Groundnut** (Oilseed / Legume)
7. **Soybean** (Major oilseed / Legume)
8. **Chickpea** (Major Rabi pulse / Bengal gram)
9. **Pigeonpea** (Major Kharif pulse / Red gram / Arhar)
10. **Rapeseed & Mustard** (Major Rabi oilseed)

Raw record count for these 10 crops: **115,338 observations**.

### 2.2 Anomaly Isolation and Data Cleaning
During exploratory validation and IQR-based analysis (`src/11_suspicious_record_validation.py`, `src/12_create_cleaned_dataset.py`), three observations were identified with physically and biologically impossible yield values:
1. **Chickpea, 2007, Muzaffarpur, Bihar:** Yield = **58,363.64 kg/ha** (~58.4 tonnes/ha; typical chickpea yield is 600–1,500 kg/ha).
2. **Chickpea, 2007, Saharsa, Bihar:** Yield = **32,000.00 kg/ha** (32.0 tonnes/ha).
3. **Pigeonpea, 2009, Dhanbad, Jharkhand:** Yield = **12,402.60 kg/ha** (12.4 tonnes/ha; typical pigeonpea yield is 700–1,200 kg/ha).

* **Action:** Excluded from the training/modeling set.
* **Cleaned Full Dataset Size:** **198,366 records** (`dataset/crop_yield_dataset_cleaned.csv`).
* **Processed 10-Crop Dataset Size:** **115,335 records** (`outputs/processed_crop_yield.csv`).

### 2.3 Distinction: Statistical Outliers vs. Erroneous Records
* Statistical outlier detection (`src/10_statistical_outlier_detection.py`) using the standard $1.5 \times \text{IQR}$ rule flagged **2,632 records (2.28%)** across the 10 crops.
* Crucially, **flagged records were NOT deleted** from the dataset. High yields occurring in well-irrigated progressive districts represent genuine agro-ecological realities. Only the three biologically impossible records above were removed.

---

## 3. Feature Space and Preprocessing Architecture

### 3.1 Feature Exclusion
Administrative and metadata columns were excluded from model input:
* `Region`, `State`, `District` (prevent memorization of administrative labels).
* `temp_src`, `rain_src`, `pest_src` (data provenance metadata).

### 3.2 Categorical One-Hot Encoding
* `Crop` (10 classes) $\rightarrow$ 10 binary dummy variables (`Crop_Chickpea`, `Crop_Cotton`, `Crop_Groundnut`, `Crop_Maize`, `Crop_Pigeonpea`, `Crop_Rapeseed & Mustard`, `Crop_Rice`, `Crop_Soybean`, `Crop_Sugarcane`, `Crop_Wheat`).
* `Season` (3 classes) $\rightarrow$ 3 binary dummy variables (`Season_Kharif`, `Season_Rabi`, `Season_Whole Year`).
* `Soil Type` (6 classes) $\rightarrow$ 6 binary dummy variables (`Soil Type_Alfisols`, `Soil Type_Aridisols`, `Soil Type_Entisols`, `Soil Type_Inceptisols`, `Soil Type_Mollisols`, `Soil Type_Vertisols`).

### 3.3 Numeric Features (10 Variables)
1. `Year` (Temporal trend / technological proxy)
2. `Temperature` (Mean growing season temperature, °C)
3. `Rainfall` (Seasonal total precipitation, mm)
4. `Humidity` (Relative humidity, %)
5. `pH` (Soil acidity/alkalinity)
6. `Nitrogen` (Available soil nitrogen, kg/ha)
7. `Phosphorus` (Available soil phosphorus, kg/ha)
8. `Potassium` (Available soil potassium, kg/ha)
9. `Fertilizer` (Chemical fertilizer application rate, kg/ha)
10. `Pesticide` (Pesticide application rate, kg/ha)

**Total Input Feature Dimension:** $10 + 10 + 3 + 6 = \mathbf{29}$ features.  
**Target Variable:** `Yield` (measured in **kg/ha**, continuous numeric regression target).

---

## 4. Chronological Train–Test Partitioning

To avoid the severe temporal data leakage inherent in random cross-validation, a chronological holdout partition was enforced (`src/04_train_test_split.py`):
* **Training Period (1966–2007, 42 years):** **92,913 observations** (80.56% of processed dataset).
* **Testing Period (2008–2017, 10 years):** **22,422 observations** (19.44% of processed dataset).

---

## 5. Machine Learning Algorithms and Scaling Protocol

Three regression algorithms were implemented (`src/05_model_training.py`):
1. **Multiple Linear Regression (LR):**
   * Implemented via `sklearn.pipeline.Pipeline` with `StandardScaler()` applied to feature inputs prior to ordinary least squares regression.
   * *Feature scaling is strictly limited to Linear Regression.*
2. **Decision Tree Regressor (DTR):**
   * Implemented using `DecisionTreeRegressor(random_state=42)`.
   * Fitted on **raw, unscaled** features. Invariant to monotonic transformations.
3. **Random Forest Regressor (RFR):**
   * Implemented using `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`.
   * Fitted on **raw, unscaled** features. Ensemble of 100 decorrelated decision trees using bootstrap aggregation.

---

## 6. Model Evaluation Results

Evaluated on the chronological test holdout (2008–2017, $N = 22,422$ records):

| Algorithm | MAE (kg/ha) | MSE ((kg/ha)²) | RMSE (kg/ha) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 769.6852 | 1,421,233.6738 | 1,192.1551 | 0.635876 |
| **Decision Tree Regressor** | 695.9377 | 1,561,169.1083 | 1,249.4675 | 0.600024 |
| **Random Forest Regressor** | **548.5934** | **901,183.2178** | **949.3067** | **0.769114** |

### Crop-Wise Random Forest Performance Breakdown
| Crop | Test Records ($N$) | MAE (kg/ha) | MSE ((kg/ha)²) | RMSE (kg/ha) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Chickpea** | 2,551 | 253.7040 | 124,180.0421 | 352.3919 | 0.182168 |
| **Cotton** | 1,403 | 164.4617 | 87,380.8684 | 295.6026 | 0.051443 |
| **Groundnut** | 2,249 | 392.7751 | 286,137.3572 | 534.9181 | 0.291885 |
| **Maize** | 2,681 | 859.8864 | 1,584,569.3264 | 1,258.7968 | 0.390511 |
| **Pigeonpea** | 2,460 | 277.0533 | 151,792.8577 | 389.6060 | 0.136536 |
| **Rapeseed & Mustard** | 2,267 | 256.4454 | 125,867.7903 | 354.7785 | 0.432694 |
| **Rice** | 2,703 | 517.4896 | 481,242.2128 | 693.7162 | 0.485341 |
| **Soybean** | 1,225 | 352.6962 | 202,788.9321 | 450.3209 | 0.068690 |
| **Sugarcane** | 2,370 | 1,582.0164 | 4,805,780.7418 | 2,192.2091 | 0.339367 |
| **Wheat** | 2,513 | 553.4408 | 508,368.8704 | 712.9999 | 0.574175 |
| **Overall** | **22,422** | **548.5934** | **901,183.2178** | **949.3067** | **0.769114** |

---

## 7. Key Diagnostic Insights and Scientific Limitations

1. **Inter-Crop vs. Within-Crop Variance:**
   * Global $R^2 = 0.7691$ is heavily inflated by distinguishing high-biomass crops (Sugarcane mean yield ~70,000 kg/ha) from field crops (Rice/Wheat ~2,000–3,500 kg/ha; Pulses/Oilseeds ~600–1,200 kg/ha).
   * Within individual crops, $R^2$ drops sharply: Chickpea (0.182), Cotton (0.051), Soybean (0.069), Pigeonpea (0.137).
2. **Concept Drift / Distribution Shift:**
   * In Cotton, mean yield jumped from 228.32 kg/ha (1966–2007) to 388.45 kg/ha (2008–2017) due to Bt cotton adoption, causing tree models to hit an extrapolation ceiling.
3. **Collinearity in Feature Importance:**
   * `Crop_Sugarcane` (0.2927) and `Season_Whole Year` (0.2923) account for 58.5% of total tree impurity reduction due to their mutual co-occurrence and massive numerical yield variance.
4. **Estimation vs. Forecasting:**
   * The pipeline performs *in-season / post-harvest empirical yield estimation* based on concurrent season weather and inputs, not *pre-season forward forecasting*.
