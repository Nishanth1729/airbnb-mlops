from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Define PROJECT_ROOT before using it
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Airbnb_Cleaned_Data.csv"
PREPROCESSOR_PATH = PROJECT_ROOT / "artifacts" / "preprocessor.pkl"


def preprocess_data():
    print(f"📂 Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Remove rows with missing prices
    print(f"Original data shape: {df.shape}")
    df = df.dropna(subset=["price"])
    print(f"After removing missing prices: {df.shape}")

    # Target variable
    y = df["price"]

    # Remove target and any potential leakage features from X
    X = df.drop(columns=["price"])

    # CRITICAL FIX: Split data BEFORE preprocessing to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Identify feature types from training data only
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")

    # Create preprocessing pipelines
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    # Fit preprocessor on TRAINING data only
    print("🔧 Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data using fitted preprocessor
    print("🔄 Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)

    # Save the preprocessor for later use in API
    import os

    os.makedirs(PROJECT_ROOT / "artifacts", exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"💾 Preprocessor saved to {PREPROCESSOR_PATH}")

    # Also save the column names and feature info for reference
    feature_names = {
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "total_features": len(categorical_cols) + len(numerical_cols),
    }
    joblib.dump(feature_names, PROJECT_ROOT / "artifacts" / "feature_names.pkl")
    print(
        f"💾 Feature info saved to {PROJECT_ROOT / 'artifacts' / 'feature_names.pkl'}"
    )

    print("✅ Data preprocessing completed")
    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        feature_names,
    )
