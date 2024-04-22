import streamlit as st
import pandas as pd 
import requests
from function import *

# Define the base URL of the API
API_BASE_URL = "http://127.0.0.1:8000"

# Function to train the model
def train_model(train_data):
    files = {'uploadedFile': ("train_data.csv", train_data)}
    response = requests.post(f"{API_BASE_URL}/train/", files=files)
    return response

# Function to make price prediction
def predict_price(prediction_data):
    files = {'input_data': ("prediction_data.csv", prediction_data)}
    response = requests.post(f"{API_BASE_URL}/predict/", files=files)
    return response

# Function to retrieve the model
def retrieve_model():
    response = requests.get(f"{API_BASE_URL}/model/")
    return response

# Main page
def main():
    st.title("Prediction of tortilla prices in Mexico")

    # Navigation bar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Documentation", "Model Training", "Price Prediction"])

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
    st.write("""
    ## API Endpoints Documentation

    This documentation provides information about the API endpoints available for the Tortilla Price Prediction App.

    ### Train Model Endpoint

    - **Description**: Trains a machine learning model using the provided training data.
    - **Method**: POST
    - **URL**: `/train/`
    - **Parameters**:
        - `data`: Training data in CSV format.
    - **Response**:
        - `message`: Indicates the status of the training process.

    ### Predict Price Endpoint

    - **Description**: Makes price predictions using the trained model.
    - **Method**: POST
    - **URL**: `/predict/`
    - **Parameters**:
        - `data`: Data for price prediction in CSV format.
    - **Response**:
        - `message`: Indicates the success of the prediction process.
        - `prediction`: Predicted prices.

    ### Retrieve Model Endpoint

    - **Description**: Retrieves the trained machine learning model.
    - **Method**: GET
    - **URL**: `/model/`
    - **Response**:
        - `model`: Trained machine learning model.
        - `coef`: Coefficients of the model.
        - `intercept`: Intercept of the model.

    ### Additional Information

    - The API base URL is `http://127.0.0.1:8000`.
    - Data should be provided in CSV format for both training and prediction endpoints.
    """)

# Model training page
def model_training_page():
    st.header("Model Training")

    # File uploader for model training data
    train_data_file = st.file_uploader("Training Data (CSV)", type=["csv"])

    if st.button("Train Model") and train_data_file is not None:
        # Read the uploaded CSV file
        train_data = pd.read_csv(train_data_file)

        # Train the model
        response = train_model(train_data)
        if response.status_code == 200:
            st.success("Model trained successfully!")
        else:
            st.error("Error during model training.")

# Price prediction page
def price_prediction_page():
    st.header("Price Prediction")

    # File uploader for price prediction data
    prediction_data_file = st.file_uploader("Prediction Data (CSV)", type=["csv"])

    if st.button("Predict Price") and prediction_data_file is not None:
        # Read the uploaded CSV file
        prediction_data = pd.read_csv(prediction_data_file)

        # Make price prediction
        response = predict_price(prediction_data)
        if response.status_code == 200:
            prediction_result = response.json()
            st.success(f"Prediction successful! Result: {prediction_result}")
        else:
            st.error("Error during prediction.")

# Retrieve model page
def retrieve_model_page():
    st.header("Retrieve Model")

    # Button to retrieve the model
    if st.button("Retrieve Model"):
        # Retrieve the model
        response = retrieve_model()
        if response.status_code == 200:
            st.success("Model retrieved successfully!")
        else:
            st.error("Error during model retrieval.")


# Exécution de l'application
if __name__ == "__main__":
    main()
