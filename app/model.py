import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import HTTPException

# ✅ Remplacer par ton vrai Run ID (visible depuis mlflow ui)
RUN_ID = "193e2732018b45d08338d95a3e5056b1"
MODEL_URI = f"runs:/{RUN_ID}/model"
SCALER_URI = f"runs:/{RUN_ID}/scaler"

try:
    svm_model = mlflow.sklearn.load_model(MODEL_URI)
    scaler = mlflow.sklearn.load_model(SCALER_URI)
    print("✅ Modèle et scaler chargés avec succès.")
except Exception as e:
    raise RuntimeError(f"❌ Erreur de chargement : {str(e)}")

def predict_class(features: list[float]):
    """Effectuer la prédiction à partir des features"""
    try:
        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        prediction = svm_model.predict(scaled_features)
        return int(prediction[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")
