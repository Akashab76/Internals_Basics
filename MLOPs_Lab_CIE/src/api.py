from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json

app = FastAPI()

# Load best model (Lasso)
model = joblib.load("models/Lasso.pkl")

# Input validation (STRICT ranges)
class InputData(BaseModel):
    order_weight_kg: float = Field(..., ge=0.5, le=15)
    distance_km: float = Field(..., ge=0.5, le=10)
    is_peak_hour: int = Field(..., ge=0, le=1)
    items_count: int = Field(..., ge=1, le=20)

@app.get("/ping")
def ping():
    return {
        "status": "running",
        "model": "Lasso",
        "version": "1.0"
    }

@app.post("/forecast")
def forecast(data: InputData):
    features = np.array([[
        data.order_weight_kg,
        data.distance_km,
        data.is_peak_hour,
        data.items_count
    ]])

    prediction = model.predict(features)[0]

    return {"prediction": float(prediction)}
