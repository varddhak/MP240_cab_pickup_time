import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.mp240.config import ROOT, load_params
from src.mp240.clean import clean_train

PARAMS = load_params()
FEATURES = PARAMS["features"]["numeric"] + PARAMS["features"]["categorical"]
MODEL_PATH = ROOT / "models" / "model.joblib"
LOG_PATH = ROOT / "reports" / "predictions.jsonl"

app = FastAPI(
    title="MP240 Cab Pickup Time API",
    description="Cab pickup ETA prediction service for Guwahati",
    version="1.0.0"
)

logger = logging.getLogger("mp240")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3)
handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

model = joblib.load(MODEL_PATH)

raw = pd.read_csv(ROOT / PARAMS["data"]["train"])
cleaned = clean_train(raw, PARAMS)
baseline = cleaned[PARAMS["features"]["numeric"]].agg(["mean", "std"])


class PredictionInput(BaseModel):
    drivers_nearby: float = Field(ge=0)
    distance_to_driver_km: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    rain_mm: float = Field(ge=0)
    surge: float = Field(ge=0)
    road_closures: int = Field(ge=0)
    car_type: str
    zone: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}


@app.get("/metadata")
def metadata():
    return {
        "project": "MP240 Cab Pickup Time",
        "target": PARAMS["data"]["target"],
        "features": FEATURES,
        "model": "Gradient Boosting champion"
    }


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        payload = data.model_dump()
        frame = pd.DataFrame([payload], columns=FEATURES)
        prediction = float(model.predict(frame)[0])

        drift = []
        for col in PARAMS["features"]["numeric"]:
            mean = baseline.loc["mean", col]
            std = baseline.loc["std", col]
            if pd.notna(std) and std > 0:
                z = abs((payload[col] - mean) / std)
                if z > 3:
                    drift.append({
                        "feature": col,
                        "z_score": round(float(z), 3)
                    })

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction_minutes": round(prediction, 3),
            "drift_detected": len(drift) > 0,
            "drift_features": drift
        }

        logger.info(json.dumps({
            "event": "prediction",
            "input": payload,
            "output": result
        }))

        return result

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
