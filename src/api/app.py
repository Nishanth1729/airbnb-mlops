import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .predictor import PricePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Airbnb Price Prediction API",
    description="API for predicting Airbnb listing prices",
    version="1.0.0",
)

predictor = None


@app.on_event("startup")
async def load_artifacts():
    global predictor
    try:
        predictor = PricePredictor()
        feature_info = predictor.get_feature_info()
        logger.info("✅ All artifacts loaded successfully")
        logger.info(f"   Categorical features: {len(feature_info['categorical_features'])}")
        logger.info(f"   Numerical features: {len(feature_info['numerical_features'])}")
        logger.info(f"   Total features: {feature_info['total_features']}")
    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {e}")
        raise


class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    version: Optional[str] = None

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
                },
                "version": "v1.0.0",
            }
        }


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
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "predictor_loaded": predictor is not None,
        "artifacts_count": 3 if predictor is not None else 0,
    }


@app.get("/features")
async def get_features():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    try:
        feature_info = predictor.get_feature_info()
        return {
            "categorical_features": feature_info["categorical_features"],
            "numerical_features": feature_info["numerical_features"],
            "total_features": feature_info["total_features"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feature info: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    try:
        prediction = predictor.predict(request.features)
        return PredictionResponse(predicted_price=float(prediction[0]))
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Feature validation error: {str(e)}. Use /features endpoint to see required features.",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
