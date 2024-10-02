import os
import joblib
import numpy as np

model_path = os.getcwd()+"rf_model.joblib"

def preprocess_features(X):
    X = np.array(X).reshape(1,-1)
    ct = joblib.load("column_transformer.joblib")
    sc = joblib.load("standard_scaler.joblib")
    
    X = ct.transform(X)
    X = sc.transform(X)
    
    return X

model = joblib.load("rf_model.joblib")