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
def predict_price(file, prediction_count):
    response = requests.post(f"{API_BASE_URL}/predict", files={"input_data": (file.name, file, file.type)}, params={"prediction_count": prediction_count})
    return response.json()

# Function to retrieve the model
def retrieve_model(question):
    response = requests.post(f"{API_BASE_URL}/model?question={question}")
    return response.json()


def show_table(file, count):
    table = pd.read_csv(file)
    data_cleaned = table.dropna()
    data_cleaned = data_cleaned.drop_duplicates()
    data_cleaned = data_cleaned.head(count)
    st.write(data_cleaned) 

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
    train_file = st.file_uploader("Upload CSV", type=["csv"], key="train")

    if st.button("Train Model") and train_file  is not None:
        try:
            # Train the model
            response = train_model(train_file)
            st.success("Model trained successfully!")
            st.success(response)
        except Exception as e:
            st.error("An error occurred:", e)


# Price prediction page
def price_prediction_page():
    st.header("Price Prediction")
    prediction_count = st.number_input("Number of predictions", min_value=1, value=100)

    # File uploader for price prediction data
    prediction_file = st.file_uploader("Prediction Data (CSV)", type=["csv"])   
    
    if prediction_file is not None:
       show_table(prediction_file, prediction_count)

    if st.button("Predict Price") and prediction_file is not None:
        try:
            # Make price prediction
            response = predict_price(prediction_file, prediction_count)
            st.success(f"Prediction successful!")
            st.success(response)  
                    
        except Exception as e:
            st.error("An error occurred")
            st.error(e)


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
