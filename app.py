import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Titanic Survival Prediction')

# Collect user inputs
pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, max_value=500.0, value=32.20)
sex = st.selectbox('Sex', ['male', 'female'])

# Convert inputs to DataFrame
input_df = pd.DataFrame({
    'PassengerId': [0],  # or any dummy/default value
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Sex_male': [1 if sex == 'male' else 0],
    'Embarked_Q': [0],   # default or user input if you want
    'Embarked_S': [0]    # default or user input if you want
})


# Apply scaler and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

# Show prediction
st.subheader('Prediction')
st.write('Survived' if prediction == 1 else 'Not Survived')
