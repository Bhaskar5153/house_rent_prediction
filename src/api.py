from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from src.modeling import predict_raw
import json


app = FastAPI()
model_path = Path(__file__).resolve().parents[1] /"artifacts"/ "model.pkl"
model = joblib.load(model_path)

@app.get("/")
def home():
    return {
        "Message": "Welcome to House rent price prediction..."
    }


df = pd.read_csv(r"C:\Users\Priya Bhaskar\OneDrive\Documents\p1_house_rent\house_rent_prediction\artifacts\processed.csv", encoding="utf-8")
features = list(model.feature_names_in_)
house = df[features].head(1)
print(json.dumps(json.loads(house.to_json(orient="records"))[0], indent=2))


@app.post("/predict")
def predict(house: dict):
    data = pd.DataFrame([house])
    data = data[features]
    prediction = model.predict(data)
    return {
        "predicted_rent": float(prediction[0])
    }

