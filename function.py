import datetime
import os
import pickle
import random
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

def init_model_training(data: pd.DataFrame):
    """
    Description de l'endpoint

    Args:
        data (): Le dataset sur lequel l'entrainement se fera.

    Returns:
        message: Message de succes de l'entrainement.
    """

    try: 
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Le dataset doit être un DataFrame")     
        
        #nettoyage et préparation des données
        cleaned_data = clean_data(data)
        
        #selection d'un echantillon de données pour l'entrainement
        cleaned_data = cleaned_data.sample(frac=0.2, random_state=1)

        X = cleaned_data.drop("Price per kilogram", axis=1)
        target_value = cleaned_data["Price per kilogram"]


        # Entrainement du modèle
        X_train, X_test, y_train, y_test = train_test_split(X, target_value, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=15, random_state=1)
        model.fit(X_train, y_train)
        
        # Evaluation du modèle
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Sauvegarde du modèle
        model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"

        if not os.path.exists("models"):
            os.makedirs("models")

        with open(f'models/' + model_name, 'wb') as in_save_file:
            pickle.dump(model, in_save_file)
        
        if os.path.exists(f'models/' + model_name):
            return {"model_name": model_name, "mse": mse}
        else:
            raise RuntimeError("Erreur lors de la sauvegarde du modèle")
        
    except Exception as e:
        return {"Error": str(e)}


def get_last_saved_model():
    """
    Recuperation des données du modele entrainé.

    Args:
       no args

    Returns:
        dict: Les données sur le modele entrainé.
    """
    try:
        models_in_dir = os.listdir("models")
        if not models_in_dir:
            raise FileNotFoundError("Aucun modèle n'est disponible.")

        models = [f'models/{model}' for model in os.listdir('models/')]
        latest_model = max(models, key=os.path.getctime)

        with open(latest_model, "rb") as file:
            model = pickle.load(file)

        return model
    except FileNotFoundError as e:
        return {"error": "Le fichier n'a pas été trouvé. Veuillez réessayer"}
    except Exception as e:
        return {"error": f"Une erreur est survenue lors de la récupération du modèle"}


def predict_price(data: pd.DataFrame, prediction_count: int = 100): 
    """
    Fonction de prédiction des prix.

    Args:
        data: Les données sur lesquelles la prédiction se fera.

    Returns:
        array: Les prix prédits.
    """ 
    try:
        model = get_last_saved_model()
        
        data_for_prediction = clean_data(data)
        
        data_for_prediction = data_for_prediction.head(prediction_count)
        
        data_for_prediction = data_for_prediction.drop("Price per kilogram", axis=1)
        
        if isinstance(data_for_prediction, dict) and "Error" in data_for_prediction:
            print("Une erreur s'est produite lors du nettoyage des données:", data_for_prediction["Error"])
            exit(1)

        prediction = model.predict(data_for_prediction)
        
        return prediction
    
    except Exception as e:
        return {"Error": str(e)}
    

def clean_data(data: pd.DataFrame):
    """
    Fonction de nettoyage des données.

    Args:
        data: Les données à nettoyer.

    Returns:
        pd.DataFrame: Les données nettoyées.
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Le dataset doit être un DataFrame")
        
        data_cleaned = data.dropna()
        data_cleaned = data_cleaned.drop_duplicates()

        numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
        data_cleaned[numeric_cols] = data_cleaned[numeric_cols].apply(lambda x: x.fillna(x.median()))

        categorical_cols = data_cleaned.select_dtypes(include=['object', 'category']).columns
        data_cleaned = pd.get_dummies(data_cleaned, columns=categorical_cols)

        return data_cleaned
    
    except Exception as e:
        return {"Error": str(e)}