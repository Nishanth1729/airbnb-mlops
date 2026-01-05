from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Define PROJECT_ROOT before using it
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'Airbnb_Cleaned_Data.csv'
PREPROCESSOR_PATH = PROJECT_ROOT / 'artifacts' / 'preprocessor.pkl'

def preprocess_data():
    print(f"📂 Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    # Remove rows with missing prices
    print(f"Original data shape: {df.shape}")
    df = df.dropna(subset=["price"])
    print(f"After removing missing prices: {df.shape}")

    # Target variables
    y_reg = df["price"]
    y_cls = pd.cut(
        df["price"],
        bins=[0, 100, 300, 1000],
        labels=["cheap", "mid", "luxury"]
    )

    X = df.drop(columns=["price"])

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Save the preprocessor for later use in API
    import os
    os.makedirs(PROJECT_ROOT / 'artifacts', exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"💾 Preprocessor saved to {PREPROCESSOR_PATH}")
    
    # Also save the column names for reference
    feature_names = {
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols
    }
    joblib.dump(feature_names, PROJECT_ROOT / 'artifacts' / 'feature_names.pkl')

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_reg, test_size=0.2, random_state=42
    )

    print("✅ Data preprocessing completed")
    return X_train, X_test, y_train, y_test