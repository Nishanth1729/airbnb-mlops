import logging
import os
import sys
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)

def train_models(X_train, y_train, preprocessor=None, feature_info=None, version=None):
    print(f"   Training RandomForest with {len(y_train)} samples...")

    params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "model_type": "RandomForestRegressor",
    }

    mlflow.set_experiment("airbnb-price-prediction")

    with mlflow.start_run():
        mlflow.log_params(params)

        model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, verbose=1
        )
        model.fit(X_train, y_train)

        mlflow.log_metric("n_train_samples", len(y_train))
        mlflow.log_metric("n_features", X_train.shape[1])
        mlflow.sklearn.log_model(model, "random_forest")

        artifact_manager = ArtifactManager()

        if preprocessor is not None and feature_info is not None:
            saved_version = artifact_manager.save_artifacts(
                model=model,
                preprocessor=preprocessor,
                feature_info=feature_info,
                metrics={},
                params=params,
                version=version,
            )
            print(f"   ✅ Model saved as version {saved_version}")
            mlflow.log_param("artifact_version", saved_version)
        else:
            MODEL_DIR = "artifacts"
            MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            print(f"   Saving model to {MODEL_PATH}...")
            saved_version = "legacy"

        print("✅ Model training completed")

    return model, saved_version
