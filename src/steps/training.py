import logging
import os
# Add utils to path
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)


def train_models(X_train, y_train, preprocessor=None, feature_info=None, version=None):
    """
    Train model with proper artifact management

    Parameters:
    -----------
    X_train, y_train : Training data
    preprocessor : Fitted preprocessor object
    feature_info : Feature information dictionary
    version : Optional version string

    Returns:
    --------
    tuple: (model, version) - Trained model and version number
    """
    print(f"   Training RandomForest with {len(y_train)} samples...")

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1, verbose=1
    )

    model.fit(X_train, y_train)

    # Initialize artifact manager
    artifact_manager = ArtifactManager()

    # Prepare metadata
    params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
        "model_type": "RandomForestRegressor",
    }

    # Save artifacts with versioning
    if preprocessor is not None and feature_info is not None:
        saved_version = artifact_manager.save_artifacts(
            model=model,
            preprocessor=preprocessor,
            feature_info=feature_info,
            metrics={},  # Will be populated after evaluation
            params=params,
            version=version,
        )
        print(f"   ✅ Model saved as version {saved_version}")
    else:
        # Fallback to old method for backward compatibility
        MODEL_DIR = "artifacts"
        MODEL_PATH = os.path.join(MODEL_DIR, "price_model.pkl")
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"   Saving model to {MODEL_PATH}...")
        saved_version = "legacy"

    print("✅ Model training completed")
    return model, saved_version
