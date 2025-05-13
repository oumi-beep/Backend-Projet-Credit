from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np
import os
import logging
from contextlib import contextmanager
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("credit-default-api")

# Performance monitoring decorator
@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    logger.info(f"{name} took {end - start:.3f} seconds")

# Define the application
app = FastAPI(
    title="Credit Default Prediction API",
    description="API for predicting credit card default risk using a pre-trained Random Forest model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model
class ClientData(BaseModel):
    LIMIT_BAL: float = Field(..., description="Credit limit")
    SEX: int = Field(..., description="Gender (1=male, 2=female)")
    EDUCATION: int = Field(..., description="Education level (1=graduate school, 2=university, 3=high school, 4=others)")
    MARRIAGE: int = Field(..., description="Marital status (1=married, 2=single, 3=others)")
    AGE: int = Field(..., description="Age in years")
    PAY_0: int = Field(..., description="Repayment status in September (-1=paid duly, 1=payment delay for one month, etc.)")
    PAY_2: int = Field(..., description="Repayment status in August")
    PAY_3: int = Field(..., description="Repayment status in July")
    PAY_4: int = Field(..., description="Repayment status in June")
    PAY_5: int = Field(..., description="Repayment status in May")
    PAY_6: int = Field(..., description="Repayment status in April")
    BILL_AMT1: float = Field(..., description="Bill amount in September")
    BILL_AMT2: float = Field(..., description="Bill amount in August")
    BILL_AMT3: float = Field(..., description="Bill amount in July")
    BILL_AMT4: float = Field(..., description="Bill amount in June")
    BILL_AMT5: float = Field(..., description="Bill amount in May")
    BILL_AMT6: float = Field(..., description="Bill amount in April")
    PAY_AMT1: float = Field(..., description="Payment amount in September")
    PAY_AMT2: float = Field(..., description="Payment amount in August")
    PAY_AMT3: float = Field(..., description="Payment amount in July")
    PAY_AMT4: float = Field(..., description="Payment amount in June")
    PAY_AMT5: float = Field(..., description="Payment amount in May")
    PAY_AMT6: float = Field(..., description="Payment amount in April")

# Define the batch prediction input model
class BatchClientData(BaseModel):
    clients: List[ClientData]

# Define the response model
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    default_status: str
    risk_factors: List[Dict[str, Any]]
    recommendations: List[str]

# Define the batch response model
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# Model and scaler paths
MODEL_PATH = r'random_forest_model.pkl'
SCALER_PATH = r'scaler.pkl'

# Global variables for model and scaler
model = None
scaler = None

# Load model and scaler on startup
@app.on_event("startup")
async def startup_event():
    global model, scaler
    try: 
        with timer("Loading scaler"):
            scaler = joblib.load(SCALER_PATH)
            logger.info("Scaler loaded successfully")
        
        with timer("Loading model"):
            model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully") 
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise RuntimeError(f"Could not load model or scaler: {e}")
    except Exception as e:
        logger.error(f"Error loading model or scaler: {e}")
        raise RuntimeError(f"Error during model initialization: {e}")

# Helper function to calculate risk factors
def calculate_risk_factors(client_data):
    risk_factors = []
    
    # Payment history risk
    payment_delay = max(client_data["PAY_0"], client_data["PAY_2"], client_data["PAY_3"])
    if payment_delay > 0:
        impact = min(1.0, payment_delay / 8)
        risk_factors.append({
            "factor": "Payment History",
            "impact": round(impact, 2),
            "description": f"Payment delay of {payment_delay} month(s)"
        })
    
    # Credit utilization risk
    utilization = client_data["BILL_AMT1"] / client_data["LIMIT_BAL"]
    if utilization > 0.5:
        impact = min(1.0, utilization)
        risk_factors.append({
            "factor": "Credit Utilization",
            "impact": round(impact, 2),
            "description": f"Using {round(utilization * 100, 1)}% of credit limit"
        })
    
    # Payment to bill ratio risk
    if client_data["BILL_AMT1"] > 0:
        payment_ratio = client_data["PAY_AMT1"] / client_data["BILL_AMT1"]
        if payment_ratio < 0.2:
            impact = min(1.0, (0.2 - payment_ratio) * 5)
            risk_factors.append({
                "factor": "Payment to Bill Ratio",
                "impact": round(impact, 2),
                "description": f"Only paying {round(payment_ratio * 100, 1)}% of bill amount"
            })
    
    # Bill amount trend risk
    if client_data["BILL_AMT1"] > client_data["BILL_AMT3"] and client_data["BILL_AMT3"] > client_data["BILL_AMT6"]:
        increase_rate = (client_data["BILL_AMT1"] - client_data["BILL_AMT6"]) / client_data["BILL_AMT6"] if client_data["BILL_AMT6"] > 0 else 0
        if increase_rate > 0.1:
            impact = min(1.0, increase_rate)
            risk_factors.append({
                "factor": "Bill Amount Trend",
                "impact": round(impact, 2),
                "description": f"Bills increasing by {round(increase_rate * 100, 1)}% over 6 months"
            })
    
    return sorted(risk_factors, key=lambda x: x["impact"], reverse=True)

# Helper function to generate recommendations
def generate_recommendations(probability, risk_factors):
    recommendations = []
    
    if probability > 0.7:
        recommendations = [
            "Reduce credit limit by 50%",
            "Require immediate minimum payment",
            "Schedule urgent follow-up call with client",
            "Consider account suspension if payment not received within 7 days"
        ]
    elif probability > 0.4:
        recommendations = [
            "Reduce credit limit by 20%",
            "Send payment reminder notifications",
            "Offer debt consolidation options",
            "Schedule follow-up call with client"
        ]
    else:
        recommendations = [
            "Monitor account activity",
            "Send courtesy payment reminder",
            "No immediate action required",
            "Review again in 30 days"
        ]
    
    # Add specific recommendations based on risk factors
    for factor in risk_factors:
        if factor["factor"] == "Payment History" and factor["impact"] > 0.5:
            recommendations.append("Implement stricter payment monitoring")
        elif factor["factor"] == "Credit Utilization" and factor["impact"] > 0.7:
            recommendations.append("Suggest credit counseling services")
        elif factor["factor"] == "Payment to Bill Ratio" and factor["impact"] > 0.6:
            recommendations.append("Offer flexible payment plan options")
    
    return recommendations[:5]  # Limit to top 5 recommendations

# Prediction endpoint for a single client
@app.post("/predict", response_model=PredictionResponse)
async def predict(client_data: ClientData):
    global model, scaler
    
    # Check if model and scaler are loaded
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([client_data.dict()])
        
        # Log input data
        logger.info(f"Received prediction request: {df.to_dict(orient='records')[0]}")
        
        with timer("Preprocessing and prediction"):
            # Scale the data
            scaled_data = scaler.transform(df)
            
            # Make prediction
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0][1]
        
        # Calculate risk factors
        risk_factors = calculate_risk_factors(client_data.dict())
        
        # Generate recommendations
        recommendations = generate_recommendations(probability, risk_factors)
        
        # Determine default status
        if probability > 0.7:
            default_status = "High Risk"
        elif probability > 0.4:
            default_status = "Medium Risk"
        else:
            default_status = "Low Risk"
        
        # Create response
        response = {
            "prediction": int(prediction),
            "probability": float(probability),
            "default_status": default_status,
            "risk_factors": risk_factors,
            "recommendations": recommendations
        }
        
        logger.info(f"Prediction result: {default_status} with probability {probability:.2f}")
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_data: BatchClientData):
    global model, scaler
    
    # Check if model and scaler are loaded
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        results = []
        
        logger.info(f"Received batch prediction request with {len(batch_data.clients)} clients")
        
        with timer(f"Batch prediction for {len(batch_data.clients)} clients"):
            for client_data in batch_data.clients:
                # Convert to DataFrame
                df = pd.DataFrame([client_data.dict()])
                
                # Scale the data
                scaled_data = scaler.transform(df)
                
                # Make prediction
                prediction = model.predict(scaled_data)[0]
                probability = model.predict_proba(scaled_data)[0][1]
                
                # Calculate risk factors
                risk_factors = calculate_risk_factors(client_data.dict())
                
                # Generate recommendations
                recommendations = generate_recommendations(probability, risk_factors)
                
                # Determine default status
                if probability > 0.7:
                    default_status = "High Risk"
                elif probability > 0.4:
                    default_status = "Medium Risk"
                else:
                    default_status = "Low Risk"
                
                # Add to results
                results.append({
                    "prediction": int(prediction),
                    "probability": float(probability),
                    "default_status": default_status,
                    "risk_factors": risk_factors,
                    "recommendations": recommendations
                })
        
        return {"predictions": results}
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    if model is None or scaler is None:
        return {"status": "error", "message": "Model or scaler not loaded"}
    return {"status": "ok", "model": "random_forest", "version": "1.0.0"}

# Model information endpoint
@app.get("/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        return {
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, "n_estimators", None),
            "feature_importances": model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else None,
            "features": list(pd.DataFrame(columns=ClientData.__annotations__.keys()).columns)
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model information: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)