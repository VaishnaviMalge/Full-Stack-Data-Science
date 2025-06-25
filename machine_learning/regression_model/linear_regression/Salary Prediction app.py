import streamlit as st
import pickle
import numpy as np

# Load the saved model
model = pickle.load(open(r"E:\Vaishnavi\practiced\machine learning\regression\linear_regression_model.pkl",'rb'))

# set the title of salary prediction app
st.title("Salary Prediction App")

# add the description
st.write("Welcome to the 'Salary Prediction App'. \nThis model predicts the salary based on year of experience using a simple regression model. ")

# Add input widget for user to enter year of experience
years_of_experience = st.number_input("Enter Years of Experience:",min_value=0.0, max_value=50.0,value =1.0, step=0.5)

# When button clicked predict Salary
if st.button("Predict Salary"):
    experience_input = np.array([[years_of_experience]])     # convert input to 2D array for prediction
    prediction = model.predict(experience_input)            # make prediction using trained model

# Display result
    st.success(f"The Predicted salary for {years_of_experience} years of experience is: ${prediction}")

# Display information about model
st.write("The model was trained using a dataset of salaries and years of experience.")

