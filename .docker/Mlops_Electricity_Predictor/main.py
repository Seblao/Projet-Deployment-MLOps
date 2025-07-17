# app/main.py

from fastapi import FastAPI
from pydantic import ValidationError
import joblib
import pandas as pd
from schema import PredictionInput

app = FastAPI(
    title="Electricity Cost Prediction API",
    description="API for predicting electricity costs from building parameters",
    version="1.0"
)

# Charger le modèle au démarrage
model = joblib.load("model.joblib")

@app.get("/")
def root():
    return {"message": "API opérationnelle. Utilisez le endpoint POST /predict pour faire une prédiction."}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Convertir l'entrée en DataFrame
        input_data = pd.DataFrame([data.dict()])
        # Prédire
        prediction = model.predict(input_data)[0]
        return {"prediction": round(prediction, 2)}
    except Exception as e:
        return {"error": str(e)}


