from fastapi import FastAPI, HTTPException
from app.schemas import FeaturesInput, PredictionOutput
from app.model import predict_class, svm_model  # Import svm_model to check feature count
import mlflow

app = FastAPI(
    title="Speech Prediction API",
    description="API for predicting speech classes using SVM model",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Speech Prediction API!"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: FeaturesInput):
    try:
        expected_features = svm_model.n_features_in_
        if len(input_data.features) != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_features} features, got {len(input_data.features)}"
            )
        prediction = predict_class(input_data.features)
        return PredictionOutput(prediction=prediction)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
