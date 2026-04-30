import streamlit as st
import pandas as pd
import joblib

# Load model + feature names
model = joblib.load("xgb_poverty_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="CA Poverty Risk Predictor", layout="centered")

st.title("California Poverty Risk Predictor")
st.write(
    "This app predicts whether an individual is likely below the poverty line "
    "based on socioeconomic indicators from the American Community Survey (ACS)."
)
st.info(
    "This tool estimates the probability that an individual falls below the U.S. poverty threshold "
    "using demographic and employment variables from the American Community Survey (ACS). "
    "Each dropdown shows the original ACS code and its description."
)
st.markdown("### Enter Individual Characteristics")

# Age
age = st.number_input(
    "Age (AGEP)",
    min_value=0,
    max_value=100,
    value=35,
    help="Age of the individual in years."
)

# Education
education = st.selectbox(
    "Education Level (SCHL)",
    options=[
        (1, "No schooling completed"),
        (16, "Regular high school diploma"),
        (19, "Some college, no degree"),
        (20, "Associate's degree"),
        (21, "Bachelor's degree"),
        (22, "Master's degree"),
        (23, "Professional degree"),
        (24, "Doctorate degree"),
    ],
    format_func=lambda x: f"{x[0]} = {x[1]}"
)[0]

# Hours worked
hours_worked = st.number_input(
    "Usual Hours Worked Per Week (WKHP)",
    min_value=0,
    max_value=99,
    value=40,
    help="Typical number of hours worked per week."
)

# Wage income
wage_income = st.number_input(
    "Annual Wage Income (WAGP)",
    min_value=0,
    value=30000,
    help="Total wage and salary income earned in the past 12 months."
)

# Employment status
employment_status = st.selectbox(
    "Employment Status (ESR)",
    options=[
        (1, "Working currently (full-time, part-time, self-employed, gig work)"),
        (2, "Has a job but temporarily not working (vacation, sick leave, parental leave, strike)"),
        (3, "Unemployed and actively looking for work"),
        (4, "Active-duty military currently working"),
        (5, "Active-duty military temporarily not working"),
        (6, "Not working and not seeking employment (student, retired, disabled, caregiver)"),
    ],
    format_func=lambda x: f"{x[0]} = {x[1]}"
)[0]

# Marital status
marital_status = st.selectbox(
    "Marital Status (MAR)",
    options=[
        (1, "Married"),
        (2, "Widowed"),
        (3, "Divorced"),
        (4, "Separated"),
        (5, "Never married / under 15"),
    ],
    format_func=lambda x: f"{x[0]} = {x[1]}"
)[0]

# Citizenship
citizenship = st.selectbox(
    "Citizenship Status (CIT)",
    options=[
        (1, "Born in the U.S."),
        (2, "Born in Puerto Rico, Guam, U.S. Virgin Islands, or Northern Marianas"),
        (3, "Born abroad to U.S. citizen parents"),
        (4, "U.S. citizen by naturalization"),
        (5, "Not a U.S. citizen"),
    ],
    format_func=lambda x: f"{x[0]} = {x[1]}"
)[0]

# Health coverage
health_coverage = st.selectbox(
    "Health Insurance Coverage (HICOV)",
    options=[
        (1, "Has health insurance coverage"),
        (2, "No health insurance coverage"),
    ],
    format_func=lambda x: f"{x[0]} = {x[1]}"
)[0]

# Class of worker
class_worker = st.selectbox(
    "Class of Worker (COW)",
    options=[
        (1, "Private for-profit employee"),
        (2, "Private not-for-profit employee"),
        (3, "Local government employee"),
        (4, "State government employee"),
        (5, "Federal government employee"),
        (6, "Self-employed in own not incorporated business"),
        (7, "Self-employed in own incorporated business"),
        (8, "Working without pay in family business"),
        (9, "Unemployed / not worked in past 5 years"),
    ],
    format_func=lambda x: f"{x[0]} = {x[1]}"
)[0]

# Build dataframe matching training structure
input_data = pd.DataFrame(columns=feature_names)
input_data.loc[0] = 0

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

input_data = input_data[feature_names]

st.markdown("### Key Drivers of Poverty Prediction")

st.write(
    "The model relies most heavily on wage income, employment status, hours worked, "
    "education level, and health insurance coverage when estimating poverty risk."
)

if st.button("Predict Poverty Risk"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Predicted: Below Poverty Line / Higher Poverty Risk")
    else:
        st.success("Predicted: Above Poverty Line / Lower Poverty Risk")

    st.write(f"Estimated probability of poverty risk: **{probability:.2%}**")
    st.write("Higher probabilities indicate greater model-estimated likelihood of being below the poverty threshold.")
    
    st.caption(
        "Prediction generated using an XGBoost classification model trained on ACS socioeconomic indicators."
    )
