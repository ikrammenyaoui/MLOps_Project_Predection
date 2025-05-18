import os
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import HTTPException

# Détecte si on tourne dans Docker (grâce à une variable d'environnement fixée dans le Dockerfile)
IN_DOCKER = os.getenv("IN_DOCKER", "0") == "1"

if IN_DOCKER:
    # Chemins locaux pour Docker (artefacts déjà copiés dans l'image)
    MODEL_PATH = "mlflow_artifacts/model"
    SCALER_PATH = "mlflow_artifacts/scaler"
    try:
        svm_model = mlflow.sklearn.load_model(MODEL_PATH)
        scaler = mlflow.sklearn.load_model(SCALER_PATH)
        print("✅ Modèle et scaler chargés depuis le dossier local (Docker).")
    except Exception as e:
        raise RuntimeError(f"❌ Erreur de chargement local : {str(e)}")
else:
    # Mode développement, chargement direct depuis le serveur MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    RUN_ID = "3175fd8203634b4bbd526dd9bc09ce79"
    MODEL_URI = f"runs:/{RUN_ID}/model"
    SCALER_URI = f"runs:/{RUN_ID}/scaler"
    try:
        svm_model = mlflow.sklearn.load_model(MODEL_URI)
        scaler = mlflow.sklearn.load_model(SCALER_URI)
        print("✅ Modèle et scaler chargés depuis MLflow (développement).")
    except Exception as e:
        raise RuntimeError(f"❌ Erreur de chargement MLflow : {str(e)}")

def predict_class(features: list[float]):
    """Effectuer la prédiction à partir des features"""
    try:
        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        prediction = svm_model.predict(scaled_features)
        return int(prediction[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {str(e)}")