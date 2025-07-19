import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = joblib.load('C:/Users/admin/Downloads/xgb_model.pkl')

# Streamlit UI
st.set_page_config(page_title="Employee Salary Classification", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Enter employee information to predict whether the salary is **>50K** or **≤50K**.")

# User Input Form
def user_input():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=75, value=30)
        education_num = st.slider("Education Level ", 4, 16, 10)
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

    with col2:

        hours_per_week = st.slider("Hours per Week", 10, 100, 40)

        marital_status = st.selectbox("Marital Status", [
            "Never-married", "Married-civ-spouse", "Divorced",
            "Separated", "Widowed", "Married-spouse-absent"
        ])

        gender = st.selectbox("Gender", ["Male", "Female"])

        occupation = st.selectbox("Occupation", [
            "Exec-managerial", "Prof-specialty", "Tech-support", "Sales", "Craft-repair",
            "Transport-moving", "Machine-op-inspct", "Adm-clerical", "Protective-serv",
            "Other-service", "Handlers-cleaners", "Farming-fishing",
            "Priv-house-serv", "Unknown"
        ])

    data = {
        'age': age,
        'educational-num': education_num,
        'hours-per-week': hours_per_week,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'marital-status': marital_status,
        'gender': gender,
        'occupation': occupation
    }
    return pd.DataFrame([data])

# Preprocessing function
def preprocess_input(df):
    # Encode categorical features
    gender_map = {'Male': 1, 'Female': 0}
    df['gender'] = df['gender'].map(gender_map)

    occupation_map = {
        "Adm-clerical": 0, "Craft-repair": 1, "Exec-managerial": 2, "Farming-fishing": 3,
        "Handlers-cleaners": 4, "Machine-op-inspct": 5, "Other-service": 6,
        "Priv-house-serv": 7, "Prof-specialty": 8, "Protective-serv": 9,
        "Sales": 10, "Tech-support": 11, "Transport-moving": 12, "Unknown": 13
    }
    df['occupation'] = df['occupation'].map(occupation_map)

    marital_map = {
        "Never-married": 0, "Married-civ-spouse": 1, "Divorced": 2,
        "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5
    }
    df['marital-status'] = df['marital-status'].map(marital_map)

    df['capital-gain'] = np.log1p(df['capital-gain'])
    df['capital-loss'] = np.log1p(df['capital-loss'])

    df['net-capital'] = df['capital-gain'] - df['capital-loss']
    df['net-capital'] = df['net-capital'].clip(lower=-5, upper=25)

    df.drop(['capital-gain', 'capital-loss'], axis=1, inplace=True)
    
    return df

# Run app
input_df = user_input()
st.write("### Input Summary", input_df)

scaler = joblib.load('C:/Users/admin/Downloads/scaler.pkl')

# Predict
if st.button("Predict"):
    input_processed = preprocess_input(input_df.copy())

    input_processed = input_processed[
        ['age', 'educational-num', 'marital-status', 'occupation',
        'gender', 'hours-per-week', 'net-capital']
    ]

    input_scaled = scaler.transform(input_processed)

    probability = model.predict_proba(input_scaled)[0][1]
    threshold = 0.438
    prediction = 1 if probability >= threshold else 0
    label = ">50K" if prediction == 1 else "≤50K"

    st.markdown(f"### 🧾 Prediction: **{label}**")
    st.markdown(f"**Probability of >50K Salary**: `{probability:.2f}`")
