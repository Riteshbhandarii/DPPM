"""FastAPI app for the final random-forest price-range estimator."""

import os
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add the repository root so local imports work when the app is launched directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.random_forest_serving import load_random_forest_bundle, predict_price_ranges


# Load the production-ready bundle trained on all available labeled data.
DEFAULT_BUNDLE_DIR = Path("artifacts/random_forest_final/full_data_bundle")
MODEL_BUNDLE = load_random_forest_bundle(os.getenv("MODEL_BUNDLE_DIR", str(DEFAULT_BUNDLE_DIR)))

app = FastAPI(
    title="DPPM Price Estimator API",
    description="Serve the final random-forest model with a model-based price range.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    """Request schema for one or more price predictions."""

    rows: list[dict[str, object]]
    lower_quantile: float = Field(default=0.10, ge=0.01, le=0.49)
    upper_quantile: float = Field(default=0.90, ge=0.51, le=0.99)


@app.get("/health")
def health_check():
    """Return a lightweight service health response."""

    return {
        "status": "ok",
        "model_type": MODEL_BUNDLE["metadata"]["model_type"],
        "bundle_split": MODEL_BUNDLE["metadata"]["bundle_split"],
        "config_name": MODEL_BUNDLE["metadata"]["config_name"],
    }


@app.get("/model-info")
def model_info():
    """Expose model metadata for quick inspection."""

    return MODEL_BUNDLE["metadata"]


@app.post("/predict")
def predict(request: PredictionRequest):
    """Return point estimates and model-based price ranges."""

    try:
        predictions = predict_price_ranges(
            MODEL_BUNDLE,
            request.rows,
            lower_quantile=request.lower_quantile,
            upper_quantile=request.upper_quantile,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "model_type": MODEL_BUNDLE["metadata"]["model_type"],
        "bundle_split": MODEL_BUNDLE["metadata"]["bundle_split"],
        "feature_variant": MODEL_BUNDLE["metadata"]["feature_variant"],
        "config_name": MODEL_BUNDLE["metadata"]["config_name"],
        "predictions": predictions.round(2).to_dict(orient="records"),
    }
