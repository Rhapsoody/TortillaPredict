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

        X = data.drop("Price per kilogram", axis=1)
        target_value = data["Price per kilogram"]

        print("training model...")

        # Entrainement du modèle
        X_train, X_test, y_train, y_test = train_test_split(X, target_value, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=20, random_state=1)
        model.fit(X_train, y_train)
        
        model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"

        print(f"Saving model to {model_name}")

        if not os.path.exists("models"):
            os.makedirs("models")

        with open(f'models/' + model_name, 'wb') as in_save_file:
            pickle.dump(model, in_save_file)
        
        if os.path.exists(f'models/' + model_name):
            return {"Training model" : "Done...", "model_name": model_name}
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


def predict_price(data: pd.DataFrame):
    """
    Fonction de prédiction des prix.

    Args:
        data: Les données sur lesquelles la prédiction se fera.

    Returns:
        array: Les prix prédits.
    """ 
    try:
        model = get_last_saved_model()

        print("last model loaded...", model)

        prediction = model.predict(data)
        return prediction
    
    except Exception as e:
        return {"Error": str(e)}