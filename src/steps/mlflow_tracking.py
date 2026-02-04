from datetime import datetime

import mlflow
import mlflow.sklearn


def setup_mlflow():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("airbnb-price-prediction")


def log_experiment(model, metrics, params):
    print("\n📊 Step 4: Logging to MLflow...")

    with mlflow.start_run(
        run_name=f"rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print("   ✅ Experiment logged to MLflow")
        print(
            "   💡 Run 'mlflow ui --backend-store-uri sqlite:///mlflow.db' to view the dashboard"
        )
