import csv
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from function import train_model, clean_df

app = FastAPI(
    title="My First FastAPI",
)

class Data(BaseModel):
    age: int
    gender: int
    platform: int
    interests: int
    location: int
    demographics: int
    profession: int
    income: int
    indebt: bool
    isHomeOwner: bool
    Owns_Car: bool

    class Config:
        json_schema_extra = {
            "example": {
                "age": 25,
                "gender": 1,
                "platform": 1,
                "interests": 2,
                "location": 3,
                "demographics": 1,
                "profession": 2,
                "income": 50000,
                "indebt": False,
                "isHomeOwner": True,
                "Owns_Car": False
            }
        }

@app.post("/training", tags=["Training"], description="Train a mode with a provided .csv as parameter")
async def train(file: UploadFile = File(...)):
    # Vérifiez si le fichier est bien un fichier CSV
    if not file.filename.endswith(".csv"):
        return JSONResponse(status_code=400, content={"error": "Seuls les fichiers CSV sont autorisés."})

    # Enregistrez le fichier CSV téléchargé
    with open("data.csv", "wb") as csv_file:
        contents = await file.read()
        csv_file.write(contents)

    # Appelez votre fonction de training avec le chemin du fichier CSV
    train_model("data.csv")

    return JSONResponse(status_code=200, content={"message": "Entraînement terminé"})

@app.post("/predict", tags=["Predict"], description="Predict user screen time with trained model and data parameter")
def predict(data:Data):
    try:
        # Charger le modèle
        model = load_model('model.h5')
    except Exception as e:
        # En cas d'erreur lors du chargement du modèle, renvoyer une erreur 500 Internal Server Error
        raise HTTPException(status_code=500, detail="Internal Server Error. Failed to load model.")

    try:
        # Prétraiter les données
        data_list = [
            data.age,
            data.gender,
            data.platform,
            data.interests,
            data.location,
            data.demographics,
            data.profession,
            data.income,
            data.indebt,
            data.isHomeOwner,
            data.Owns_Car
        ]

        # Convertir les booléens en entiers
        data_list[8:] = [int(value) for value in data_list[8:]]

        # Faire une prédiction
        prediction = model.predict([data_list])

        # Renvoyer la prédiction
        return {"Prediction": prediction.tolist()}
    except Exception as e:
        # En cas d'erreur lors de la prédiction, renvoyer une erreur 500 Internal Server Error
        raise HTTPException(status_code=500, detail="Internal Server Error. Failed to make prediction.")

# Route pour télécharger le fichier model.h5
@app.get("/download-model", tags=["Download"])
async def download_model():
    try:
        # Chemin vers le fichier model.h5
        model_path = "model.h5"
        # Retourne le fichier comme une réponse de fichier
        return FileResponse(model_path, media_type="application/octet-stream", filename="model.h5")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to download model file.")

# Ajoutez une route pour rediriger toutes les erreurs 404 vers /docs
@app.middleware("http")
async def redirect_404_to_docs(request: Request, call_next):
    response = await call_next(request)
    if response.status_code == 404:
        return RedirectResponse(url="/docs")
    return response