#API ENTRAINEMENT DE MODELE ET PREDICTION

from fastapi import FastAPI


app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API de prédiction des prix de tortillas au Mexique..."}


@app.post("/train/")
async def train_model(data: dict):
    """
    Description de l'endpoint

    Args:
        item_id (int): L'identifiant de l'élément à récupérer.
        q (str, optional): Un paramètre de requête optionnel.

    Returns:
        dict: Les données de l'élément récupéré.
    """
    return {"message": "Succès de l'entrainement..."}


@app.post("/predict/")
async def predict_price(data: dict):
    return {"message": "Prédiction succès..."}


@app.get("/model/")
async def get_model():
    return {"message": "Récupération du modèle..."} 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
