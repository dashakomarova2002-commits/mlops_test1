from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Titanic Survival Prediction API")

# Загружаем модель и scaler при старте приложения
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


# Описываем структуру входных данных (как в датасете Титаника)
class PassengerFeatures(BaseModel):
    Pclass: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float


@app.post("/predict")
def predict(features: PassengerFeatures):
    # Преобразуем входной JSON в DataFrame для scaler
    input_df = pd.DataFrame([features.model_dump()])

    # Масштабируем данные
    input_scaled = scaler.transform(input_df)

    # Делаем предсказание
    prediction = model.predict(input_scaled)

    # Возвращаем результат
    return {"prediction": int(prediction[0])}