import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.artifact_manager import ArtifactManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PricePredictor:
    def __init__(self, version: str = None):
        try:
            artifact_manager = ArtifactManager()

            if version:
                logger.info(f"Loading artifacts for version {version}...")
                self.model, self.preprocessor, self.feature_info, self.metadata = (
                    artifact_manager.load_latest_artifacts()
                )
            else:
                logger.info("Loading latest artifacts...")
                self.model, self.preprocessor, self.feature_info, self.metadata = (
                    artifact_manager.load_latest_artifacts()
                )

            logger.info("✅ All artifacts loaded successfully")

            if self.metadata:
                logger.info(f"   Model version: {self.metadata.get('version', 'unknown')}")
                logger.info(f"   Model type: {self.metadata.get('model_type', 'unknown')}")

                total_features = self.feature_info.get("total_features")
                if total_features is None:
                    total_features = len(self.feature_info.get("categorical_cols", [])) + len(
                        self.feature_info.get("numerical_cols", [])
                    )

                logger.info(f"   Features: {total_features}")

        except Exception as e:
            logger.error(f"❌ Failed to load artifacts: {e}")
            raise

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        required_features = set(
            self.feature_info["categorical_cols"] + self.feature_info["numerical_cols"]
        )
        input_features = set(df.columns)

        missing_features = required_features - input_features
        extra_features = input_features - required_features

        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        if extra_features:
            logger.warning(f"Extra features will be ignored: {extra_features}")

        expected_order = self.feature_info["categorical_cols"] + self.feature_info["numerical_cols"]
        return df[expected_order]

    def predict(self, features):
        try:
            if isinstance(features, dict):
                features = pd.DataFrame([features])

            validated_features = self._validate_features(features)
            processed_features = self.preprocessor.transform(validated_features)
            prediction = self.model.predict(processed_features)

            return prediction

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_single(self, **kwargs) -> float:
        prediction = self.predict(kwargs)
        return float(prediction[0])

    def get_feature_info(self) -> dict:
        categorical = self.feature_info.get("categorical_cols", [])
        numerical = self.feature_info.get("numerical_cols", [])

        total = self.feature_info.get("total_features")
        if total is None:
            total = len(categorical) + len(numerical)

        return {
            "categorical_features": categorical,
            "numerical_features": numerical,
            "total_features": total,
            "model_version": self.metadata.get("version", "unknown") if self.metadata else "unknown",
            "model_type": self.metadata.get("model_type", "unknown") if self.metadata else "unknown",
        }
