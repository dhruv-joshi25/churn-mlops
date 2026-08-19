import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api import predict as engine
from src.api.models import (
    BatchRequest,
    BatchResponse,
    Customer,
    Health,
    Prediction,
)

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_model()  # warm start; failure is non-fatal, /health reports it
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    version="0.1.0",
    description="Serves the churn model with SHAP-based reasons and retention hints.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any public deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=Health, tags=["ops"])
def health():
    loaded = engine.is_loaded()
    return Health(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=engine.version() if loaded else None,
        threshold=engine.threshold() if loaded else None,
        threshold_source=engine.threshold_source() if loaded else None,
    )


@app.post("/reload", tags=["ops"])
def reload_model():
    """Pull the current Production model without restarting the container."""
    model = engine.load_model(force=True)
    if model is None:
        raise HTTPException(503, "no model available to load")
    return {
        "reloaded": True,
        "model_version": engine.version(),
        "threshold": engine.threshold(),
        "threshold_source": engine.threshold_source(),
    }


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict_one(customer: Customer, explain: bool = True):
    try:
        return engine.predict([customer.model_dump()], with_reasons=explain)[0]
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(req: BatchRequest, explain: bool = False):
    try:
        rows = [c.model_dump() for c in req.customers]
        return BatchResponse(predictions=engine.predict(rows, with_reasons=explain))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
