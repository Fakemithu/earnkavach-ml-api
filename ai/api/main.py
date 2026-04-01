from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# load model
model = joblib.load("../models/income_model.pkl")

@app.get("/")
def home():
    return {"message": "EarnKavach ML API Running"}

@app.post("/predict-income")
def predict(data: dict):
    features = np.array([[
        data["day_of_week"],
        data["hour_of_day"],
        data["avg_last_7_days"],
        data["rainfall"],
        data["aqi"],
        data["temperature"],
        data["zone_demand_score"],
        data["is_weekend"],
        data["working_hours"]
    ]])

    prediction = model.predict(features)[0]

    return {"predicted_income": float(prediction)}