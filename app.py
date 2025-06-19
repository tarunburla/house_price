import numpy as np
import sklearn as sk
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Title
st.title("🏠 House Price Prediction App")

# Sidebar Inputs
st.sidebar.header("Enter House Features")

# User input fields
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 10, 2)
sqft = st.sidebar.number_input("Square Feet", 500, 10000, 1500)
location_score = st.sidebar.slider("Location Score (1-10)", 1, 10, 5)

# Prepare the input data
input_data = pd.DataFrame({
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms],
    'Sqft': [sqft],
    'Location': [location_score]
})

# Dummy training data
X = pd.DataFrame({
    'Bedrooms': [2, 3, 4, 5, 6],
    'Bathrooms': [1, 2, 2, 3, 4],
    'Sqft': [1000, 1500, 2000, 2500, 3000],
    'Location': [5, 6, 7, 8, 9]
})

# Prices (target variable)
y = [100000, 150000, 200000, 250000, 300000]

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Make prediction
prediction = model.predict(input_data)

# Display result
st.subheader("Predicted House Price:")
st.write(f"${prediction[0]:,.2f}")
