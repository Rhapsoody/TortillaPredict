import datetime
import os
import pickle
import random
import pandas as pd

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

        X = data["State", "City", "Year", "Month", "Day", "Store type"]
        target_value = data["Price per kilogram"]


        # Entrainement du modèle
        X_train, X_test, y_train, y_test = train_test_split(X, target_value, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=20, random_state=1)
        model.fit(X_train, y_train)
        
        model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
        with open(f'models/' + model_name, 'wb') as in_save_file:
            pickle.dump(model, in_save_file)
        
        return {"Training model" : "Done..."}
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
    models_in_dir = os.listdir("models")
    if not models_in_dir:
        return {"error": "Aucun modèle n'est disponible."}

    latest_model = max(models_in_dir, key=os.path.getctime)
    with open(f"models/{latest_model}", "rb") as file:
        model = pickle.load(file)

    return {"model": model, "coef": model.coef_, "intercept": model.intercept_}    


def predict_price(data: pd.DataFrame):
    """
    Fonction de prédiction des prix.

    Args:
        data: Les données sur lesquelles la prédiction se fera.

    Returns:
        array: Les prix prédits.
    """ 
    try:
        model = get_last_saved_model()["model"]
        prediction = model.predict(data)
        return {"Price per kilogram": prediction}
    except Exception as e:
        return {"Error": str(e)}