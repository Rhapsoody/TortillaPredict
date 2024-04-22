import streamlit as st
import pandas as pd 
import requests
from function import *

# Define the base URL of the API
API_BASE_URL = "http://127.0.0.1:8000"

# Function to train the model
def train_model(file):
    response = requests.post(f"{API_BASE_URL}/train", files={"uploadedFile": (file.name, file, file.type)})    
    return response.json()

# Function to make price prediction
def predict_price(file):
    response = requests.post(f"{API_BASE_URL}/predict", files={"input_data": (file.name, file, file.type)})
    return response.json()

# Function to retrieve the model
def retrieve_model(question):
    response = requests.post(f"{API_BASE_URL}/model?question={question}")
    return response.json()

# Main page
def main():
    st.title("Prediction of tortilla prices in Mexico")

    # Navigation bar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Documentation", "Model Training", "Price Prediction", "Retrieve Model"])

    # Display the selected page
    if page == "Documentation":
        home_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Price Prediction":
        price_prediction_page()
    elif page == "Retrieve Model":
        retrieve_model_page()

# Home page
def home_page():
    st.title("Documentation")
    st.components.v1.iframe("http://127.0.0.1:8000/docs#/default", width=800, height=2000, scrolling=True)

# Model training page
def model_training_page():
    st.header("Model Training")

    # File uploader for model training data
    train_file = st.file_uploader("Upload CSV",type=["csv"], key="train")

    if st.button("Train Model") and train_file  is not None:
        try:
            # Train the model
            response = train_model(train_file)
            if response.get('status_code') == 200:
                st.success(f"{response.get('message')}")
                st.success("Model : " +f"{response.get('model')}")
            else:
                st.error(response.get('message'))
        except Exception as e:
            st.error("An error occurred:", e)


# Price prediction page
def price_prediction_page():
    st.header("Price Prediction")

    # File uploader for price prediction data
    prediction_file = st.file_uploader("Prediction Data (CSV)", type=["csv"])

    if st.button("Predict Price") and prediction_file is not None:
        try:
            # Make price prediction
            response = predict_price(prediction_file)
            if response.get('status_code') == 200:
                st.success(f"Prediction successful! Result: {response.get('predictions')}")
            else:
                st.error("Error during prediction:", response)
        except Exception as e:
            st.error("An error occurred:", e)


# Retrieve model page
def retrieve_model_page():
    st.header("Retrieve Model")
    question = st.text_input('Enter a question')

    # Button to retrieve the model
    if st.button("Retrieve Model"):
        # Retrieve the model
        response = retrieve_model(question)
        if response.get('status_code') == 200:
            st.success(response.get('message'))
        else:
            st.error("Error during model retrieval.")


# Exécution de l'application
if __name__ == "__main__":
    main()
