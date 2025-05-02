import joblib
import numpy as np
import os
from fastapi import HTTPException

def get_model_path(filename):
    """Get absolute path to model files"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    return os.path.join(models_dir, filename)

# Load model and scaler (make them global)
try:
    svm_model = joblib.load(get_model_path('svm_model_updated.pkl'))
    scaler = joblib.load(get_model_path('scaler_updated.pkl'))
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

def predict_class(features: list[float]):
    """Predict class from input features"""
    try:
        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        prediction = svm_model.predict(scaled_features)
        return int(prediction[0])
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )