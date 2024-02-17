from pydantic import BaseModel
from typing import Dict

class InputIris(BaseModel):
    petal_lenght : float
    petal_width: float




class OutputPredict(BaseModel):
    results : Dict[str, float]

    