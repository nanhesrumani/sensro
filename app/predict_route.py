from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class SensorInput(BaseModel):
    speed:            Optional[float] = None
    accel_x:          Optional[float] = None
    accel_y:          Optional[float] = None
    accel_z:          Optional[float] = None
    accel_magnitude:  Optional[float] = None
    gyro_x:           Optional[float] = None
    gyro_y:           Optional[float] = None
    gyro_z:           Optional[float] = None
    gyro_magnitude:   Optional[float] = None
    accuracy:         Optional[float] = None
    altitude:         Optional[float] = None
    heading:          Optional[float] = None

@router.post("/predict")
async def predict(sensor: SensorInput):
    try:
        from ml.predict import predict_condition
        result = predict_condition(sensor.dict())
        return result
    except FileNotFoundError as e:
        return {"error": str(e), "condition": None}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "condition": None}

@router.get("/predict/model-info")
async def model_info():
    try:
        import os, json
        root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        best_path = os.path.join(root, "ml", "models", "best_model.txt")
        csv_path  = os.path.join(root, "ml", "model_comparison.csv")

        if not os.path.exists(best_path):
            return {"error": "No model trained yet"}

        with open(best_path) as f:
            best = f.read().strip()

        comparison = []
        if os.path.exists(csv_path):
            import csv
            with open(csv_path) as f:
                comparison = list(csv.DictReader(f))

        return {"best_model": best, "comparison": comparison}
    except Exception as e:
        return {"error": str(e)}
