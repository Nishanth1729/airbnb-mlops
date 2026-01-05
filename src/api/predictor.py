import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / "artifacts" / "price_model.pkl"

class PricePredictor:
    def __init__(self):
        print(f"Loading model from {MODEL_PATH}...")
        self.model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    
    def predict(self, features):
        """
        Make price predictions
        
        Parameters:
        -----------
        features : dict or pd.DataFrame
            Input features for prediction
        
        Returns:
        --------
        float or array : Predicted price(s)
        """
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        prediction = self.model.predict(features)
        return prediction
    
    def predict_single(self, **kwargs):
        """Predict price for a single listing"""
        prediction = self.predict(kwargs)
        return float(prediction[0])

# Example usage
if __name__ == "__main__":
    predictor = PricePredictor()
    
    # Example prediction (adjust features based on your data)
    sample_data = {
        # Add your actual features here
        # This is just an example structure
    }
    
    price = predictor.predict_single(**sample_data)
    print(f"\n💰 Predicted Price: ${price:,.2f}")