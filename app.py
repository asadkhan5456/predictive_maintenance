from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Predictive Maintenance API")

# Load the trained model from disk
model = joblib.load("model/model.pkl")

# Define the input data schema for prediction
class PredictRequest(BaseModel):
    op_setting1: float
    op_setting2: float
    op_setting3: float
    sensor_avg: float

@app.post("/predict")
def predict(data: PredictRequest):
    # Convert incoming data into the format expected by the model
    features = np.array([[data.op_setting1, data.op_setting2, data.op_setting3, data.sensor_avg]])
    prediction = model.predict(features)
    return {"predicted_RUL": prediction[0]}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Predictive Maintenance API. Use the /predict endpoint to get RUL predictions."}
