# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Loan Default Prediction API", 
              description="API for predicting loan default risk",
              version="1.0.0")

# Load improved model
try:
    model = joblib.load('models/improved_model.pkl')
    feature_info = joblib.load('models/feature_info.pkl')
    MODEL_LOADED = True
except:
    MODEL_LOADED = False

class LoanApplication(BaseModel):
    annual_inc: float
    loan_amnt: float  
    fico_score: int
    dti: float
    experience: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    recommendation: str

@app.get("/")
async def root():
    return {"message": "Loan Default Prediction API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": MODEL_LOADED}

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create features in correct order
        features = np.array([[
            application.annual_inc,    # revenue
            application.dti,           # dti_n
            application.loan_amnt,     # loan_amnt  
            application.fico_score,    # fico_n
            application.experience     # experience_c
        ]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Determine risk level and recommendation
        if probability < 0.3:
            risk_level = "low"
            recommendation = "Approve"
        elif probability < 0.6:
            risk_level = "medium" 
            recommendation = "Review further"
        else:
            risk_level = "high"
            recommendation = "Reject"
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)