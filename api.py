#API ENTRAINEMENT DE MODELE ET PREDICTION

from fastapi import FastAPI
import pickle

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import function as fn

import pandas as pd

from openai import OpenAI

app = FastAPI()

# TODO: Repasser sur les commentaires



@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API de prédiction des prix de tortillas au Mexique..."}


@app.post("/train", tags=["Training/Prediction"])
async def train_model(uploadedFile: UploadFile = File(...)):
    """
    Point de terminaison d'entrainement du modele de prédiction des prix.

    Args:
        uploadedFile (UploadFile, optional): Le fichier CSV contenant les données d'entrainement.

    Returns:
        dict: Les données de l'élément récupéré.
    """
    if uploadedFile is None:
        raise HTTPException(status_code=400, detail="Aucun fichier n'a été envoyé.")
    
    if uploadedFile.filename.split(".")[-1] != "csv":
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas un fichier CSV.")
    
    data = pd.read_csv(uploadedFile.file)

    if data is None:
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas un fichier CSV.")

    try:
        model = fn.init_model_training(data)

        return {"message": "Entrainement du modèle réussi...", "model": model}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})


@app.post("/predict", tags=["Training/Prediction"])
async def predict_price(input_data: UploadFile = File(...)):

    if input_data is None:
        raise HTTPException(status_code=400, detail="Aucun fichier n'a été envoyé.")

    if input_data.filename.split(".")[-1] != "csv":
        raise HTTPException(status_code=400, detail="Le fichier envoyé n'est pas un fichier CSV.")
    
    data = pd.read_csv(input_data.file)

    if data is None:
        raise HTTPException(status_code=400, detail="Echec de lecture du fichier CSV.")
    
    try:
        prediction_input = data.drop("Price per kilogram", axis=1)

        prediction_output = fn.predict_price(prediction_input)

        return {"predictions": prediction_output.tolist()}
    
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": str(e)})


@app.post("/model", tags=["OpenAI"])
async def question_answering(question: str):
    """
    Endpoint faisant appel à l'API de OpenAI pour répondre à une question.

    Args:
        question (str): La question à laquelle on veut une réponse.

    Returns:
        dict: La réponse à la question posée.
    """

    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")
    
    client = OpenAI(
        
        api_key=""
    )

    try:
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model="gpt-3.5-turbo",
        )

        response = chat_completion.choices[0].message.content

        return {"success => reponse =": response}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
