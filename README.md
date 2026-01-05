# Airbnb Price Prediction – AI Engineer–Grade MLOps Pipeline

## 📌 Project Overview
This repository implements a **production-oriented Machine Learning system** for predicting Airbnb listing prices using structured features such as location, room type, and availability.

The project is intentionally designed from an **AI Engineer perspective**, focusing on how models are:
- Trained as reproducible pipelines
- Packaged as deployable artifacts
- Served through scalable, low-latency APIs
- Integrated into real-world applications

Unlike notebook-driven prototypes, this system reflects **industry ML deployment workflows**.

---

## 🧠 AI Engineering Focus
This project demonstrates core competencies expected from an **AI Engineer**:

- **Model-as-a-Service Design** (REST-based inference)
- **Separation of Training and Serving Pipelines**
- **Stateless API Inference**
- **Artifact-Based Model Versioning**
- **Production-Ready Input Validation**
- **Deployment via Containerization**

---

## 🚀 Key Features
- **High-Performance Inference API**  
  Built with **FastAPI**, providing schema-enforced requests using Pydantic and auto-generated OpenAPI documentation.

- **Decoupled Training Pipeline**  
  Model training is isolated from serving logic, enabling independent retraining and redeployment.

- **Containerized Deployment**  
  Docker ensures consistency across development, staging, and production environments.

- **Modular & Maintainable Codebase**  
  Clear separation between API layer, ML logic, and data schemas.

- **Reproducible ML Workflow**  
  Dependency control via `requirements.txt` and serialized model artifacts.

---

## 🛠️ Technology Stack
- **Programming Language:** Python 3.9+
- **ML Frameworks:** Scikit-learn, Pandas, NumPy
- **API & Serving:** FastAPI, Uvicorn
- **Data Validation:** Pydantic
- **Containerization:** Docker
- **Tooling:** Git, Postman

---

## 📂 Repository Structure
airbnb-mlops/
├── app/
│   ├── main.py            # FastAPI application entry point
│   ├── model.py           # Model loading and inference logic
│   └── schemas.py         # Pydantic request/response schemas
├── model/
│   ├── train.py           # Offline model training pipeline
│   └── saved_model.pkl    # Serialized model artifact
├── Dockerfile             # Production-ready container configuration
├── requirements.txt       # Dependency specification
└── README.md              # Project documentation
