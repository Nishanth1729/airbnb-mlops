from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Initialize FastAPI
app = FastAPI(
    title="Airbnb Price Prediction API",
    description="API for predicting Airbnb listing prices",
    version="1.0.0"
)

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_PATH / "artifacts" / "price_model.pkl"
PREPROCESSOR_PATH = BASE_PATH / "artifacts" / "preprocessor.pkl"
FEATURE_NAMES_PATH = BASE_PATH / "artifacts" / "feature_names.pkl"

# Global variables
model = None
preprocessor = None
feature_info = None

@app.on_event("startup")
async def load_artifacts():
    global model, preprocessor, feature_info
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        feature_info = joblib.load(FEATURE_NAMES_PATH)
        print("✅ All artifacts loaded successfully")
        print(f"   Categorical features: {feature_info['categorical_cols']}")
        print(f"   Numerical features: {feature_info['numerical_cols']}")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

# Request model - accepts raw features as a dictionary
class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "accommodates": 4,
                    "bedrooms": 2.0,
                    "beds": 2.0,
                    "bathrooms": 1.5,
                    "property_type": "Apartment",
                    "room_type": "Entire home/apt",
                    # Add all other features from your dataset
                }
            }
        }

# Response model
class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"

@app.get("/")
async def root():
    return {
        "message": "🏠 Airbnb Price Prediction API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "features": "/features",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.get("/features")
async def get_features():
    """Get the list of features required for prediction"""
    if feature_info is None:
        raise HTTPException(status_code=503, detail="Feature info not loaded")
    
    return {
        "categorical_features": feature_info['categorical_cols'],
        "numerical_features": feature_info['numerical_cols'],
        "total_features": len(feature_info['categorical_cols']) + len(feature_info['numerical_cols'])
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict Airbnb listing price"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")
    
    try:
        # Convert features dict to DataFrame
        df = pd.DataFrame([request.features])
        
        # Apply preprocessing
        X_processed = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(X_processed)
        
        return PredictionResponse(
            predicted_price=float(prediction[0])
        )
    
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required feature: {str(e)}. Use /features endpoint to see required features."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)