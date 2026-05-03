
# California Poverty Risk Prediction Using Machine Learning

## Project Overview

This project builds a machine learning model to predict whether an individual falls below the poverty threshold using demographic and employment characteristics from the American Community Survey (ACS). The goal is to demonstrate how predictive analytics can support early identification of economically vulnerable populations and assist policymakers in allocating resources more effectively.

An interactive deployment of the final model is available here:

**Streamlit App:**  
https://predicting-ca-poverty-aziza-fong-marcela.streamlit.app/

---

## Research Questions

### Primary Research Question

Which individuals in California are most vulnerable to poverty based on their education level, employment conditions, demographic characteristics, and access to health insurance?

### Supporting Research Questions

#### Education and Poverty Risk  
How does education level affect the likelihood of experiencing poverty in California?

#### Employment Conditions and Poverty Risk 
Which employment patterns are most associated with poverty risk?

#### Demographic Inequality and Poverty Risk

How do age, marital status, citizenship status, and other demographic characteristics relate to poverty vulnerability?

---
### Data Download and Subset Creation

The dataset is derived from the **American Community Survey (ACS) Public Use Microdata Sample (PUMS)**, which provides demographic, employment, and income indicators across the United States. Because the full ACS PUMS file is very large and contains hundreds of variables, we created a focused project subset for faster processing and reproducibility.

Using DuckDB, we selected variables related to education, employment, income, demographics, health insurance, and disability status.
The final subset contains 25,000 sampled California individual records and 27 modeling variables. This subset was saved as a CSV and uploaded to GitHub so the notebook can be run without repeatedly downloading and processing the full ACS file.

### Features Used

- Age (AGEP)
- Education Level (SCHL)
- Employment Status (ESR)
- Hours Worked Per Week (WKHP)
- Wage Income (WAGP)
- Citizenship Status (CIT)
- Health Insurance Coverage (HICOV)
- Class of Worker (COW)
- Marital Status (MAR)

### Target Variable

- Poverty Status (Binary Classification)

---

## Why Poverty Prediction Matters

Timely identification of individuals at risk of poverty helps improve allocation of public assistance programs and policy interventions. Machine learning methods allow analysts to estimate poverty risk using demographic indicators when direct income measurements may be incomplete or delayed.

---

## Methods

Three classification models were trained and compared:

- Logistic Regression
- Random Forest
- XGBoost (Final Selected Model)

### Evaluation Metric

Models were evaluated using **ROC–AUC**, which measures how well the model distinguishes poverty versus non-poverty outcomes across classification thresholds. AUC was selected because it is robust to class imbalance and provides strong ranking performance.

---

## Final Model Selection

XGBoost achieved the strongest predictive performance and was selected as the final model.

Tree-based ensemble methods such as Random Forest and XGBoost are well suited for poverty prediction because they capture nonlinear relationships between demographic variables and economic outcomes.

---

## Key Predictors of Poverty Risk

The model identified the following variables as most influential:

- Wage Income (WAGP)
- Employment Status (ESR)
- Hours Worked Per Week (WKHP)
- Education Level (SCHL)
- Health Insurance Coverage (HICOV)

These predictors align with established research showing income stability, employment participation, and access to services strongly influence poverty classification outcomes.

---

## Streamlit Deployment

An interactive prediction interface was built using Streamlit to allow users to estimate poverty risk based on individual characteristics.

The deployed app:

- accepts ACS demographic inputs
- explains Census category codes
- returns poverty classification results
- displays estimated probability of poverty risk

Access the live application here:

**https://predicting-ca-poverty-aziza-fong-marcela.streamlit.app/**

---

## Example Use Case

A policymaker or analyst can input:

- employment status
- education level
- weekly working hours
- wage income
- insurance coverage

to estimate whether an individual is likely below the poverty threshold and prioritize intervention strategies accordingly.

---

## Model Assumptions

### Logistic Regression

Assumes:

- binary outcome variable
- linear relationship between predictors and log-odds
- low multicollinearity
- independent observations
- sufficient sample size

All assumptions were satisfied.

### Random Forest

Assumes:

- nonlinear feature relationships allowed
- no distributional assumptions
- robustness to correlated predictors

These conditions were appropriate for the ACS dataset structure.

### XGBoost

Assumes:

- nonlinear predictor interactions
- sufficient dataset size
- hyperparameter tuning to reduce overfitting

These conditions were satisfied through model optimization.

---

## Limitations

This analysis relies on cross-sectional ACS survey data rather than longitudinal income trajectories. Poverty classification is income-based and does not capture multidimensional deprivation such as housing stability or food security.

Future work could incorporate geographic cost-of-living adjustments and additional social indicators.

---

## Technologies Used

- Python
- pandas
- scikit-learn
- XGBoost
- Streamlit
- joblib

---

## Repository Structure

```
MLFinalProject.ipynb
app.py
requirements.txt
xgb_poverty_model.pkl
feature_names.pkl
ca_poverty_subset_final.csv
```

---

## Social Impact

Accurate poverty prediction models help identify vulnerable populations earlier and support more effective allocation of housing, employment, and healthcare resources. Machine learning tools can complement traditional survey methods to improve poverty monitoring and intervention planning.
