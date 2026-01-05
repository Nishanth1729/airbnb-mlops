from sklearn.ensemble import RandomForestRegressor
import joblib
import os

MODEL_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")

def train_models(X_train, y_train):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"   Training RandomForest with {len(y_train)} samples...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_train)
    
    print(f"   Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    print("✅ Model training completed")
    return model  # Return the trained model




