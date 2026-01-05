from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import json
import os

METRICS_DIR = "artifacts"
METRICS_PATH = os.path.join(METRICS_DIR, "metrics.json")

def evaluate_model(model, X_test, y_test):
    print("\n📈 Step 3: Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2_score": float(r2)
    }
    
    # Save metrics
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Display metrics
    print(f"\n   Model Performance:")
    print(f"   ├─ MAE (Mean Absolute Error): ${mae:,.2f}")
    print(f"   ├─ RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    print(f"   └─ R² Score: {r2:.4f}")
    
    print(f"\n   ✅ Metrics saved to {METRICS_PATH}")
    return metrics