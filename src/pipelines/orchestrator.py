import logging
import os

import joblib

from src.steps.evaluation import evaluate_model
from src.steps.mlflow_tracking import log_experiment, setup_mlflow
from src.steps.preprocessing import preprocess_data
from src.steps.training import train_models
from src.utils.artifact_manager import ArtifactManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline():
    print("🚀 Starting Airbnb MLOps Pipeline\n")

    # Setup MLflow
    setup_mlflow()

    # Step 1: Preprocess data
    print("📊 Step 1: Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor, feature_info = preprocess_data()
    print(f"   Training set shape: X={X_train.shape}, y={y_train.shape}")

    # Step 2: Train model
    print("\n🤖 Step 2: Starting model training...")
    print("   (This may take 2-5 minutes...)")
    model, version = train_models(X_train, y_train, preprocessor, feature_info)

    # Step 3: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Step 4: Update artifact metadata with metrics
    artifact_manager = ArtifactManager()
    try:
        # Update the version with metrics
        version_dir = artifact_manager.versions_dir / version
        if version_dir.exists():
            import json

            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            metadata["metrics"] = metrics
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"   ✅ Updated version {version} with evaluation metrics")
    except Exception as e:
        logger.warning(f"Could not update version metadata: {e}")

    # Step 5: Log to MLflow
    params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "model_type": "RandomForestRegressor",
        "version": version,
    }
    log_experiment(model, metrics, params)

    print("\n🎯 Pipeline completed successfully!")
    print("\n" + "=" * 50)
    print("📁 Artifacts saved:")
    print(f"   ├─ Model version: {version}")
    print(f"   ├─ Latest: artifacts/latest/")
    print(f"   └─ Versioned: artifacts/versions/{version}/")
    print(f"   └─ Metrics: artifacts/metrics.json")
    print("\n💡 Next steps:")
    print("   1. Run 'mlflow ui' to view experiment tracking")
    print("   2. Run 'python -m src.api.app' to start the API")
    print("   3. Visit http://localhost:8000/docs for API documentation")
    print("   4. Check artifacts/versions/ for model history")
    print("=" * 50)

    return model, version, metrics


if __name__ == "__main__":
    run_pipeline()
