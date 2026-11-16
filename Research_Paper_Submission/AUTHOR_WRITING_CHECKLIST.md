# Author Writing Checklist & AI-Limit Compliance Guide

**Project Title:** Crop Yield Prediction in India Using Machine Learning Algorithms  
**Target Submission Format:** Springer Journal / Conference Template  
**Target Author:** B.Tech Computer Science & Engineering Undergraduate Student  
**AI Content Requirement:** Maximum 10% AI-generated text permitted by academic advisor.

---

## 1. How to Use the Prepared Research Paper Draft

The prepared document (`Crop_Yield_Prediction_Research_Paper.docx`) provides the verified mathematical formulas, exact empirical result tables, figure captions, methodology descriptions, and peer-reviewed literature citations.

To ensure your submission contains **less than 10% AI-generated text**, you must write the original research discussion, domain reflections, and personal analytical synthesis in your own words. 

In the DOCX document, look for callout boxes marked:
`[STUDENT AUTHOR DISCUSSION NOTE: ...]`

These indicate where you should replace the guidance prompts with your original writing. Use the checklist below to guide your writing.

---

## 2. Section-by-Section Original Writing Checklist

### Section 1: Introduction
- [ ] **Problem Motivation (Paragraph 2):** In your own words, describe why crop yield uncertainty affects smallholder farmers in India. Mention monsoon dependence, crop failure risks, or market price volatility.
- [ ] **Personal Motivation (Paragraph 4):** Write a brief sentence explaining why you chose this topic for your B.Tech final project (e.g., combining data science with a real-world national challenge).
- [ ] **Scope Limitation (Paragraph 6):** Explicitly clarify that this project evaluates historical empirical yield estimation at the district level using concurrent seasonal weather and input factors, rather than real-time pre-season forecasting.

### Section 2: Background and Related Work
- [ ] **Synthesis of Prior Work (Section 2.2):** Paraphrase the findings of Indian agricultural studies (e.g., Gandhi et al., Medar et al.). Explain why simple linear regression is inadequate for agricultural systems with non-linear weather thresholds.
- [ ] **Critique of Random Train–Test Splitting (Section 2.4):** In your own words, explain why splitting historical agricultural data randomly across years causes data leakage, and why your chronological split (1966–2007 vs. 2008–2017) is a more realistic test.

### Section 3: Data and Methods
- [ ] **District Cleaning Reflection (Section 3.3):** Explain the logic behind removing the three biologically impossible records (Muzaffarpur and Saharsa Chickpea with 58 and 32 tonnes/ha; Dhanbad Pigeonpea with 12 tonnes/ha). Contrast this with the decision to keep statistical outliers flagged by IQR that represent real progressive farming.
- [ ] **Feature Selection Reflection (Section 3.4):** Explain why administrative columns (`Region`, `State`, `District`) were deliberately removed so that the models learn biophysical relationships rather than district-name memorization.
- [ ] **Streamlit Web Application Note (Section 3.7):** Write a short paragraph describing how you implemented the user interface in `app.py`, allowing users to input soil type, weather values, and crop selection to obtain instant predictions.

### Section 4: Performance Evaluation Metrics
- [ ] **Metric Interpretation (Section 4.4):** Explain why R² should never be referred to as "accuracy percentage." Write 2–3 sentences explaining why MAE is preferred for average error in physical units (kg/ha), while RMSE highlights large occasional errors.

### Section 5: Experimental Results and Scientific Discussion (CRITICAL SECTION FOR ORIGINAL WRITING)
- [ ] **Comparative Overview (Section 5.1):** Discuss why Random Forest outperformed Linear Regression and Decision Trees on the test set. Emphasize that while Random Forest won, it is not a perfect solution.
- [ ] **The "Sugarcane Dominance" Discussion (Section 5.2 & 5.5):** In your own words, explain why `Crop_Sugarcane` and `Season_Whole Year` dominate the feature importance (~58.5%). Explain that Sugarcane produces 60–100 tonnes/ha compared to 1–3 tonnes/ha for wheat and rice, dominating the variance in the target variable.
- [ ] **The "Weak Crop" Failure Analysis (Section 5.3):**
  - Discuss why Cotton had an R² of only 0.051 and Soybean had an R² of 0.069.
  - Explain the impact of the Green Revolution and Bt cotton adoption after 2002: average cotton yields rose from 228 kg/ha (1966–2007) to 388 kg/ha (2008–2017).
  - Explain why tree-based models failed to predict these higher yields because they cannot extrapolate beyond the maximum values seen in the training data.
- [ ] **Honest Limitations (Section 5.7):** Write about data limitations in your own words: lack of micro-irrigation data, pest infestation timing, real-time heatwave waves, and seed variety differences across districts.

### Section 6: Conclusion and Future Work
- [ ] **Concluding Summary (Section 6.1):** Summarize the main lesson from your B.Tech project: that machine learning can capture broad patterns across different crops, but predicting year-to-year yield variations within specific rainfed crops remains challenging without temporal time-series architectures and remote sensing data.
- [ ] **Future Recommendations (Section 6.3):** Suggest 2–3 practical future improvements, such as testing LSTM/Transformer models for temporal sequences, integrating satellite NDVI data, or training separate crop-specific models rather than a single pooled model.

---

## 3. Fact and Citation Verification Checklist

Before final submission to your teacher, verify each item below:

| Verification Item | Verified Project Value | Checked (Y/N) |
| :--- | :--- | :---: |
| Total observations in raw dataset | 198,369 rows | [ ] |
| Total observations after 3 removed | 198,366 rows | [ ] |
| Total observations for 10 selected crops | 115,335 rows | [ ] |
| Training set observations (1966–2007) | 92,913 rows (80.56%) | [ ] |
| Testing set observations (2008–2017) | 22,422 rows (19.44%) | [ ] |
| Number of input features | 29 features | [ ] |
| Linear Regression Test MAE | 769.6852 kg/ha | [ ] |
| Linear Regression Test RMSE | 1192.1551 kg/ha | [ ] |
| Linear Regression Test R² | 0.635876 | [ ] |
| Decision Tree Test MAE | 695.9377 kg/ha | [ ] |
| Decision Tree Test RMSE | 1249.4675 kg/ha | [ ] |
| Decision Tree Test R² | 0.600024 | [ ] |
| Random Forest Test MAE | 548.5934 kg/ha | [ ] |
| Random Forest Test RMSE | 949.3067 kg/ha | [ ] |
| Random Forest Test R² | 0.769114 | [ ] |
| Cotton Test R² | 0.051443 | [ ] |
| Soybean Test R² | 0.068690 | [ ] |
| Chickpea Test R² | 0.182168 | [ ] |
| Pigeonpea Test R² | 0.136536 | [ ] |
| Wheat Test R² | 0.574175 | [ ] |
| Sugarcane Feature Importance | 0.292691 (29.27%) | [ ] |
| Season_Whole Year Feature Importance | 0.292326 (29.23%) | [ ] |
| Three removed records verified | Muzaffarpur Chickpea, Saharsa Chickpea, Dhanbad Pigeonpea | [ ] |
| Standard scaler verified | Used ONLY in Linear Regression pipeline; Decision Tree & RF unscaled | [ ] |

---

## 4. Academic Tone Guidelines for Student Writing

1. **Avoid Overselling:** Do not claim that your model "revolutionizes agriculture" or "guarantees optimal crop yields." State calmly that Random Forest provides an effective baseline for cross-crop estimation, but exhibits notable limitations under decadal yield shifts.
2. **Never Call R² "Accuracy":** R² measures the proportion of variance explained by the model relative to a naive mean baseline. Calling it "76.9% accuracy" is mathematically incorrect and easily flagged by faculty reviewers.
3. **Use Consistent Units:** Always write yield as **kg/ha** (kilograms per hectare) or state metric equivalents clearly.
4. **Distinguish Estimation from Forecasting:** Estimation uses recorded weather throughout the season; pre-season forecasting predicts yield months before weather occurs. Your models perform post-season / in-season empirical estimation.
