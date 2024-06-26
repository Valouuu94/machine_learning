from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Union
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

app = FastAPI(
    title="Machine Learning API",
    description="API for training and predicting machine learning models using FastAPI",
    version="1.0.0"
)

# Define the model path
MODEL_PATH = "model.joblib"

class TrainData(BaseModel):
    features: List[List[float]]
    labels: List[Union[int, float]]

class PredictData(BaseModel):
    features: List[List[float]]

@app.get("/", summary="Welcome Message")
async def read_root():
    return {"message": "Welcome to the Machine Learning API. Go to /docs for the API documentation."}

@app.post("/training", summary="Train a model with provided data")
async def train_model(data: TrainData):
    # Convert the input data to a DataFrame
    df = pd.DataFrame(data.features)
    df['target'] = data.labels
    
    # Split the data
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model (e.g., Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, MODEL_PATH)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {"message": "Model trained successfully", "accuracy": accuracy}

@app.post("/predict", summary="Predict using the trained model")
async def predict(data: PredictData):
    # Load the model
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not found. Train a model first."}
    
    model = joblib.load(MODEL_PATH)
    
    # Convert the input data to a DataFrame
   
