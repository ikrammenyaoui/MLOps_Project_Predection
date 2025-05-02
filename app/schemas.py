from pydantic import BaseModel
from typing import List

class FeaturesInput(BaseModel):
    features: List[float]  # No default values

class PredictionOutput(BaseModel):
    prediction: int