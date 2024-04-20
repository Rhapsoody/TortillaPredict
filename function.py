import datetime
import os
import pickle
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


#TODO: Refaire les commentaires qui sont en template

def init_model_training(data):
    """
    Description de l'endpoint

    Args:
        item_id (int): L'identifiant de l'élément à récupérer.
        q (str, optional): Un paramètre de requête optionnel.

    Returns:
        dict: Les données de l'élément récupéré.
    """
    X = data["State", "City", "Year", "Month", "Day", "Store type"]
    y = data["Price per kilogram"]


    # Entrainement du modèle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    if not os.path.exists("models"):
        os.makedirs("models")

    # Sauvegarde du modèle avec un nom unique
    model_name = f"model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    with open(model_name, "wb") as file:
        pickle.dump(model, file)
    
    return {"Training model" : "Done..."}


def get_model():
    """
    Description de l'endpoint

    Args:
        item_id (int): L'identifiant de l'élément à récupérer.
        q (str, optional): Un paramètre de requête optionnel.

    Returns:
        dict: Les données de l'élément récupéré.
    """
    models = os.listdir("models")
    model_name = random.choice(models)
    with open(f"models/{model_name}", "rb") as file:
        model = pickle.load(file)
    return {"model": model_name, "coef": model.coef_, "intercept": model.intercept_}    


def predict_price(data):
    """
    Description de l'endpoint

    Args:
        item_id (int): L'identifiant de l'élément à récupérer.
        q (str, optional): Un paramètre de requête optionnel.

    Returns:
        dict: Les données de l'élément récupéré.
    """
    model = get_model()
    prediction = model.predict(data)
    return {"Price per kilogram": prediction}