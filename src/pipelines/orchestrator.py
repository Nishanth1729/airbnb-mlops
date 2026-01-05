from src.steps.preprocessing import preprocess_data
from src.steps.training import train_models
from src.steps.evaluation import evaluate_model
from src.steps.mlflow_tracking import setup_mlflow, log_experiment
import joblib
import os

def run_pipeline():
    print("🚀 Starting Airbnb MLOps Pipeline\n")
    
    # Setup MLflow
    setup_mlflow()
    
    # Step 1: Preprocess data
    print("📊 Step 1: Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data()
    print(f"   Training set shape: X={X_train.shape}, y={y_train.shape}")
    
    # Step 2: Train model
    print("\n🤖 Step 2: Starting model training...")
    print("   (This may take 2-5 minutes...)")
    model = train_models(X_train, y_train)
    
    # Step 3: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 4: Log to MLflow
    params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "model_type": "RandomForestRegressor"
    }
    log_experiment(model, metrics, params)
    
    print("\n🎯 Pipeline completed successfully!")
    print("\n" + "="*50)
    print("📁 Artifacts saved:")
    print(f"   ├─ Model: artifacts/price_model.pkl")
    print(f"   └─ Metrics: artifacts/metrics.json")
    print("\n💡 Next steps:")
    print("   1. Run 'mlflow ui' to view experiment tracking")
    print("   2. Run 'python -m src.api.app' to start the API")
    print("   3. Visit http://localhost:8000/docs for API documentation")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()
