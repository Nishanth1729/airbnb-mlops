import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
FEATURES_ENDPOINT = f"{API_BASE_URL}/features"
