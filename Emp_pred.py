import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load('C:/Users/admin/Downloads/xgb_model.pkl')

# Set page config
st.set_page_config(page_title="Salary Prediction App", layout="centered")

# Title
st.title("Employee Salary Prediction")

st.markdown("Enter employee features to predict their salary category (High or Low)")

# User input form
def user_input():
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    education_num = st.slider("Education Level (Numeric)", 1, 16, 10)
    hours_per_week = st.slider("Hours per Week", 1, 100, 40)
    marital_status = st.selectbox(
        "Marital-status",
        options=["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-AF-spouse", "Married-spouse-absent"]
    )
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"]
    )
    workclass = st.selectbox(
        "Workclass",
        options=[
            "Federal-gov", "Local-gov", "Private",
            "Self-emp-inc", "Self-emp-not-inc", "State-gov", "nan"
        ]
    )
    occupation = st.selectbox(
        "Occupation",
        options=[
            "Adm-clerical", "Craft-repair", "Exec-managerial", "Farming-fishing",
            "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv",
            "Prof-specialty", "Protective-serv", "Sales", "Tech-support",
            "Transport-moving", "nan"
        ]
    )
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 100000, 0)

    data = {
        'age': age,
        'education-num': education_num,
        'hours-per-week': hours_per_week,
        'marital-status': marital_status,
        'gender': gender,
        'occupation': occupation,
        'workclass': workclass,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
    }
    features = pd.DataFrame([data])
    return features

input_df = user_input()

def preprocess_input(df):
    # Encode gender
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Encode marital-status
    marital_map = {
        "Divorced": 0,
        "Married-AF-spouse": 1,
        "Married-civ-spouse": 2,
        "Married-spouse-absent": 3,
        "Never-married": 4,
        "Separated": 5,
        "Widowed": 6
    }
    df['marital-status'] = df['marital-status'].map(marital_map)

    # Workclass encoding
    df['workclass'] = df['workclass'].map({
        "Federal-gov": 0,
        "Local-gov": 1,
        "Private": 2,
        "Self-emp-inc": 3,
        "Self-emp-not-inc": 4,
        "State-gov": 5,
        "nan": 6
    })

    # Occupation encoding
    df['occupation'] = df['occupation'].map({
        "Adm-clerical": 0,
        "Craft-repair": 1,
        "Exec-managerial": 2,
        "Farming-fishing": 3,
        "Handlers-cleaners": 4,
        "Machine-op-inspct": 5,
        "Other-service": 6,
        "Priv-house-serv": 7,
        "Prof-specialty": 8,
        "Protective-serv": 9,
        "Sales": 10,
        "Tech-support": 11,
        "Transport-moving": 12,
        "nan": 13
    })

    return df

input_df = preprocess_input(input_df)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    label = ">50k" if prediction == 1 else "<=50k"
    
    st.markdown(f"### Prediction: **{label}**")
    st.write(f"Probability of high salary: **{prob:.2f}**")