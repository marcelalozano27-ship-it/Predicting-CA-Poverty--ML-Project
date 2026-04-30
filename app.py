import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("xgb_poverty_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="CA Poverty Risk Predictor", layout="centered")

st.title("California Poverty Risk Predictor")
st.write(
    "This app uses an XGBoost machine learning model trained on ACS data "
    "to predict whether an individual may be below the poverty line."
)

st.markdown("### Enter Individual Characteristics")

# Main numeric inputs
age = st.number_input("Age", min_value=0, max_value=100, value=35)
education = st.number_input("Education Level Code (SCHL)", min_value=0, max_value=30, value=16)
hours_worked = st.number_input("Usual Hours Worked Per Week (WKHP)", min_value=0, max_value=99, value=40)
wage_income = st.number_input("Wage Income (WAGP)", min_value=0, value=30000)

# Categorical ACS code inputs
employment_status = st.selectbox(
    "Employment Status (ESR)",
    options=[1, 2, 3, 4, 5, 6],
    help="ACS employment status code"
)

marital_status = st.selectbox(
    "Marital Status (MAR)",
    options=[1, 2, 3, 4, 5],
    help="ACS marital status code"
)

citizenship = st.selectbox(
    "Citizenship Status (CIT)",
    options=[1, 2, 3, 4, 5],
    help="ACS citizenship status code"
)

health_coverage = st.selectbox(
    "Has Health Coverage (HICOV)",
    options=[1, 2],
    help="ACS code: 1 = has coverage, 2 = no coverage"
)

class_worker = st.selectbox(
    "Class of Worker (COW)",
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    help="ACS class of worker code"
)

# Create blank row with all training features
input_data = pd.DataFrame(columns=feature_names)
input_data.loc[0] = 0

# Fill only columns that exist in training data
user_values = {
    "AGEP": age,
    "SCHL": education,
    "WKHP": hours_worked,
    "WAGP": wage_income,
    "ESR": employment_status,
    "MAR": marital_status,
    "CIT": citizenship,
    "HICOV": health_coverage,
    "COW": class_worker,
}

for col, value in user_values.items():
    if col in input_data.columns:
        input_data.loc[0, col] = value

# Reorder columns exactly like training
input_data = input_data[feature_names]

st.markdown("---")

if st.button("Predict Poverty Risk"):
    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
    else:
        probability = None

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("Predicted: Below Poverty Line / Higher Poverty Risk")
    else:
        st.success("Predicted: Above Poverty Line / Lower Poverty Risk")

    if probability is not None:
        st.write(f"Estimated probability of poverty risk: **{probability:.2%}**")

    st.caption(
        "Note: This model is for educational purposes and should not be used as the sole basis for policy or eligibility decisions."
    )
