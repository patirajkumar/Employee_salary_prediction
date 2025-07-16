## Employee Salary Prediction App
This project is a machine learning web application built with Streamlit that predicts whether an employee's salary is likely to be greater than 50K or less than or equal to 50K, based on various demographic and work-related features.

--------
## Features
- User-friendly interface built with Streamlit

- Predicts salary class: >50K or <=50K

- Displays probability confidence

- Built using a trained XGBoost classifier

- Handles categorical inputs via label encoding

- Real-time prediction on user input
--------
---------

## Model Information
Algorithm used: XGBoost Classifier

Model Selection: Chosen after comparing Logistic Regression, Random Forest, and Gradient Boosting

Accuracy: ~88%

ROC AUC Score: ~0.93

---------
---------

## Input Features
Feature	Description
Age	Integer (between 18 and 65)
Education Level	Numeric (1 to 16)
Hours per Week	Weekly working hours
Marital Status	Categorical (e.g., Married, Divorced)
Gender	Male or Female
Workclass	Type of employer
Occupation	Job role
Capital Gain	Investment gain (numeric)
Capital Loss	Investment loss (numeric)

-----------

## Built With
Python

Streamlit

XGBoost

scikit-learn

pandas

joblib

----------
