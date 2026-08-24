import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from churnkit import config
from churnkit.reference.api import predict as engine
from churnkit.reference.api.models import (
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

# An allowlist rather than "*", so a browser will not let an arbitrary page
# read this API's responses. It is also load-bearing for the /reload guard
# below: that guard works by requiring a custom header, and a custom header
# is only expensive to forge because the preflight it triggers lands here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", config.ADMIN_HEADER],
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
def reload_model(
    admin: str | None = Header(default=None, alias=config.ADMIN_HEADER),
):
    """Pull the current Production model without restarting the container.

    The only endpoint here that changes server state, and therefore the only
    one worth a cross-site request. The required header is not a credential —
    anyone who can reach the port can send it — it is there because a browser
    will not attach a custom header to a cross-origin request without first
    passing the CORS preflight, which an unlisted origin fails. That closes
    the drive-by case: a page the operator is merely visiting cannot swap the
    model underneath them. See ADR 0004 for what this deliberately does not
    defend against.
    """
    if admin is None:
        raise HTTPException(
            403,
            f"/reload changes which model is served, so it requires the "
            f"{config.ADMIN_HEADER} header. Send it with any value.",
        )
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
