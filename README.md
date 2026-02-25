# Porto Seguro Claim Prediction
### Insurance Claim Risk Prediction: A P&C Modeling Pipeline

Predicting auto insurance claim probability using machine learning on the Porto Seguro Safe Driver Prediction dataset. Built a full modeling pipeline including EDA, feature engineering, and gradient boosting models with early stopping, achieving a Gini coefficient of 0.28.

---

## Overview

This project builds an end-to-end binary classification pipeline to predict the probability that an auto insurance policyholder will file a claim in the next year. The dataset comes from the [Porto Seguro Safe Driver Prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction) Kaggle competition.

The project was built as a portfolio piece targeting a Predictive Modeler / Data Scientist role in the P&C insurance industry.

---

## Results

| Model | AUC | Gini |
|-------|-----|------|
| Logistic Regression | 0.6211 | 0.2422 |
| Random Forest | 0.6274 | 0.2549 |
| LightGBM | 0.6370 | 0.2739 |
| XGBoost | 0.6396 | 0.2792 |
| LightGBM + Early Stopping | 0.6389 | 0.2779 |
| **XGBoost + Early Stopping** | **0.6402** | **0.2804** |
| Stacked Ensemble | 0.6394 | 0.2787 |

**Best model: XGBoost with Early Stopping — Gini 0.2804**

---

## Project Structure

```
porto-seguro-claim-prediction/
│
├── Porto_Seguro_Claim_Prediction.ipynb   # Main notebook
├── README.md                              
└── submission.csv                         # Final predictions
```

---

## Pipeline

**1. EDA**
- Class imbalance analysis (96/4 split)
- Missing value identification (-1 encoding)
- Feature type separation (binary, categorical, continuous)
- Correlation analysis and feature distribution plots

**2. Preprocessing**
- Dropped `ps_calc_*` features (documented as noise)
- Dropped high-missingness columns (`ps_car_03_cat`, `ps_car_05_cat`)
- Imputed missing values with mode (categorical) and median (continuous)
- Label encoded categorical features
- Dropped 4 near-zero importance features identified by Random Forest

**3. Modeling**
- Logistic Regression baseline
- Random Forest with grid search
- LightGBM with grid search and early stopping
- XGBoost with grid search and early stopping
- Stacked ensemble (LR + RF + LightGBM + XGBoost)

**4. Class Imbalance**
- Handled via `scale_pos_weight=26.44` (ratio of negative to positive cases)
- No resampling applied in this version

---

## Key Findings

- Gradient boosting methods significantly outperformed linear and tree ensemble baselines
- Early stopping found the optimal number of trees automatically — XGBoost stopped at ~150 iterations out of a possible 2000
- Stacking did not improve on XGBoost alone, suggesting the models were too correlated to benefit from ensembling
- `ps_car_13` was the strongest single predictor, consistent across all models

---

## Tech Stack

- Python 3
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- LightGBM
- XGBoost
- Google Colab (T4 GPU)

---

## Next Steps

- Rerun pipeline with SMOTE oversampling and compare against class weighting approach
- Add missing value count as an engineered feature
- Second round of hyperparameter fine-tuning with narrower grid
- Deploy final model as a Streamlit app

---

## Author

**Alex Jurcich**
