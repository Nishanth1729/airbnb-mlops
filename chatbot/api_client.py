import requests
from config import FEATURES_ENDPOINT, PREDICT_ENDPOINT


def get_required_features(timeout: int = 20):
    r = requests.get(FEATURES_ENDPOINT, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    categorical = data.get("categorical_features", [])
    numerical = data.get("numerical_features", [])
    all_features = categorical + numerical

    return categorical, numerical, all_features


def predict_price(features: dict, timeout: int = 30):
    payload = {"features": features, "version": None}
    r = requests.post(PREDICT_ENDPOINT, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()
