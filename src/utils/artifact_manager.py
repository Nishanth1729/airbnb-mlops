import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ArtifactManager:
    """Manages model versioning and artifact storage"""

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.versions_dir = self.artifacts_dir / "versions"
        self.latest_dir = self.artifacts_dir / "latest"
        self.metadata_file = self.artifacts_dir / "metadata.json"

        # Create directories
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.latest_dir.mkdir(parents=True, exist_ok=True)

    def get_next_version(self) -> str:
        """Get next version number based on existing versions"""
        if not self.versions_dir.exists():
            return "v1.0.0"

        existing_versions = [d.name for d in self.versions_dir.iterdir() if d.is_dir()]
        if not existing_versions:
            return "v1.0.0"

        # Sort versions and increment patch version
        latest_version = sorted(existing_versions)[-1]
        major, minor, patch = map(int, latest_version[1:].split("."))
        patch += 1
        return f"v{major}.{minor}.{patch}"

    def save_artifacts(
        self,
        model: Any,
        preprocessor: Any,
        feature_info: Dict[str, Any],
        metrics: Dict[str, float],
        params: Dict[str, Any],
        version: Optional[str] = None,
    ) -> str:
        """
        Save model artifacts with versioning

        Returns:
        --------
        str: The version number of saved artifacts
        """
        if version is None:
            version = self.get_next_version()

        version_dir = self.versions_dir / version
        version_dir.mkdir(exist_ok=True)

        logger.info(f"Saving artifacts to version {version}")

        # Save artifacts to version directory
        import joblib

        joblib.dump(model, version_dir / "price_model.pkl")
        joblib.dump(preprocessor, version_dir / "preprocessor.pkl")
        joblib.dump(feature_info, version_dir / "feature_names.pkl")

        # Save metadata
        metadata = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "params": params,
            "model_type": "RandomForestRegressor",
            "feature_count": feature_info.get("total_features", 0),
        }

        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Update latest symlink
        self._update_latest(version)

        # Update global metadata
        self._update_global_metadata(metadata)

        logger.info(f"✅ Artifacts saved as version {version}")
        return version

    def _update_latest(self, version: str):
        """Update latest symlink to point to new version"""
        latest_model = self.latest_dir / "price_model.pkl"
        latest_preprocessor = self.latest_dir / "preprocessor.pkl"
        latest_features = self.latest_dir / "feature_names.pkl"
        latest_metadata = self.latest_dir / "metadata.json"

        version_dir = self.versions_dir / version

        # Remove existing files
        for file_path in [
            latest_model,
            latest_preprocessor,
            latest_features,
            latest_metadata,
        ]:
            if file_path.exists():
                file_path.unlink()

        # Create symlinks (or copy on Windows)
        try:
            latest_model.symlink_to(version_dir / "price_model.pkl")
            latest_preprocessor.symlink_to(version_dir / "preprocessor.pkl")
            latest_features.symlink_to(version_dir / "feature_names.pkl")
            latest_metadata.symlink_to(version_dir / "metadata.json")
        except OSError:
            # Fallback to copy if symlinks not supported
            shutil.copy2(version_dir / "price_model.pkl", latest_model)
            shutil.copy2(version_dir / "preprocessor.pkl", latest_preprocessor)
            shutil.copy2(version_dir / "feature_names.pkl", latest_features)
            shutil.copy2(version_dir / "metadata.json", latest_metadata)

    def _update_global_metadata(self, new_metadata: Dict[str, Any]):
        """Update global metadata file"""
        global_metadata = {}

        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                global_metadata = json.load(f)

        global_metadata["latest_version"] = new_metadata["version"]
        global_metadata["last_updated"] = new_metadata["timestamp"]

        if "versions" not in global_metadata:
            global_metadata["versions"] = []

        # Add new version if not already present
        version_info = {
            "version": new_metadata["version"],
            "timestamp": new_metadata["timestamp"],
            "metrics": new_metadata["metrics"],
            "model_type": new_metadata["model_type"],
        }

        # Remove duplicate if exists
        global_metadata["versions"] = [
            v
            for v in global_metadata["versions"]
            if v["version"] != new_metadata["version"]
        ]
        # Add new version
        global_metadata["versions"].append(version_info)
        # Sort by timestamp
        global_metadata["versions"].sort(key=lambda x: x["timestamp"], reverse=True)

        with open(self.metadata_file, "w") as f:
            json.dump(global_metadata, f, indent=4)

    def load_latest_artifacts(self):
        """Load the latest version of artifacts"""
        if not self.latest_dir.exists():
            raise FileNotFoundError("No artifacts found")

        import joblib

        model_path = self.latest_dir / "price_model.pkl"
        preprocessor_path = self.latest_dir / "preprocessor.pkl"
        feature_info_path = self.latest_dir / "feature_names.pkl"

        if not all(
            p.exists() for p in [model_path, preprocessor_path, feature_info_path]
        ):
            raise FileNotFoundError("Missing artifact files")

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        feature_info = joblib.load(feature_info_path)

        # Load metadata
        metadata_path = self.latest_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

        return model, preprocessor, feature_info, metadata

    def get_version_info(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific version or latest"""
        if version is None:
            metadata_path = self.latest_dir / "metadata.json"
        else:
            metadata_path = self.versions_dir / version / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for version {version}")

        with open(metadata_path, "r") as f:
            return json.load(f)

    def list_versions(self) -> list:
        """List all available versions"""
        if not self.versions_dir.exists():
            return []

        versions = []
        for version_dir in sorted(self.versions_dir.iterdir()):
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    versions.append(metadata)

        return versions
