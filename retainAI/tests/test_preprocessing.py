import pytest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def test_column_transformer_shape():
    # Create a tiny DataFrame
    df = pd.DataFrame({
        'num1': [10, 20, np.nan],
        'num2': [1.0, 2.0, 3.0],
        'cat1': ['a', 'b', 'a']
    })
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', RobustScaler())]), ['num1', 'num2']),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))]), ['cat1'])
    ])
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape == (3, 3)  # 2 numerical + 1 categorical after encoding? depends on encoder

def test_model_output_shape():
    import joblib
    model = joblib.load("Artifacts/07_12_2026_23_58_53/model_trainer/trained_model/model.pkl")
    # Create a random input of the correct shape
    # This is just a demo; real test should use actual preprocessor output
    dummy_input = np.random.rand(1, 10)  # adjust to actual feature count
    pred = model.predict(dummy_input)
    assert pred.shape == (1,)