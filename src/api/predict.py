"""Model loading and inference.

Loading order:
  1. MLflow model registry at the configured stage (production path)
  2. A local artifact directory (CI, offline dev)

The loaded object must be an sklearn Pipeline that accepts a raw DataFrame
with the columns in schema.FEATURES. That is the contract with the ML side.
"""

import logging
import threading

import pandas as pd

from src import config
from src.schema import FEATURES

log = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None
_explainer = None
_version = "unloaded"


def _try_registry():
    import mlflow

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    uri = f"models:/{config.REGISTERED_MODEL_NAME}/{config.MODEL_STAGE}"
    model = mlflow.sklearn.load_model(uri)
    return model, f"{config.REGISTERED_MODEL_NAME}@{config.MODEL_STAGE}"


def _try_local():
    import mlflow

    model = mlflow.sklearn.load_model(config.LOCAL_MODEL_PATH)
    return model, "local"


def load_model(force: bool = False):
    global _model, _explainer, _version
    with _lock:
        if _model is not None and not force:
            return _model
        for loader in (_try_registry, _try_local):
            try:
                _model, _version = loader()
                _explainer = None
                log.info("loaded model: %s", _version)
                return _model
            except Exception as exc:  # noqa: BLE001
                log.warning("%s failed: %s", loader.__name__, exc)
        _version = "unloaded"
        return None


def is_loaded() -> bool:
    return _model is not None


def version() -> str:
    return _version


def _frame(payloads: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(payloads)
    # TotalCharges is nullable in the request; the pipeline imputes it.
    for col in FEATURES:
        if col not in df.columns:
            df[col] = None
    return df[FEATURES]


def _get_explainer():
    """Built lazily — SHAP import is heavy and only needed on /explain."""
    global _explainer
    if _explainer is not None:
        return _explainer
    import shap

    pre = _model.named_steps["preprocess"]
    est = _model.named_steps["model"]
    try:
        _explainer = shap.TreeExplainer(est)
    except Exception:  # noqa: BLE001 — linear/other models
        _explainer = shap.Explainer(est)
    _explainer._feature_names = list(pre.get_feature_names_out())
    return _explainer


def explain(df: pd.DataFrame, top_k: int = 4) -> list[list[dict]]:
    model = load_model()
    if model is None:
        return [[] for _ in range(len(df))]
    try:
        explainer = _get_explainer()
        Xt = model.named_steps["preprocess"].transform(df)
        values = explainer.shap_values(Xt)
        names = explainer._feature_names
        out = []
        for row in values:
            ranked = sorted(
                zip(names, row), key=lambda kv: abs(kv[1]), reverse=True
            )[:top_k]
            out.append(
                [
                    {
                        "feature": n,
                        "contribution": round(float(v), 4),
                        "direction": "increases_risk" if v > 0 else "decreases_risk",
                    }
                    for n, v in ranked
                ]
            )
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning("explanation failed: %s", exc)
        return [[] for _ in range(len(df))]


def risk_band(p: float) -> str:
    if p >= 0.66:
        return "High"
    if p >= 0.33:
        return "Medium"
    return "Low"


def recommend(payload: dict, prob: float) -> str | None:
    """Placeholder for the retention logic Sanjida owns.

    Kept behind this function so her rules drop in without touching the API.
    Note these are hypotheses to A/B test, not proven causal levers.
    """
    if prob < config.DECISION_THRESHOLD:
        return None
    if payload.get("Contract") == "Month-to-month":
        return "Offer a one-year contract with a discounted monthly rate."
    if payload.get("TechSupport") == "No" and payload.get("InternetService") != "No":
        return "Offer complimentary tech support for three months."
    if (payload.get("MonthlyCharges") or 0) > 80:
        return "Review the plan for a cheaper tier matching actual usage."
    return "Route to the retention team for a loyalty offer."


def predict(payloads: list[dict], with_reasons: bool = True) -> list[dict]:
    model = load_model()
    if model is None:
        raise RuntimeError("model not available")

    df = _frame(payloads)
    probs = model.predict_proba(df)[:, 1]
    reasons = explain(df) if with_reasons else [[] for _ in payloads]

    results = []
    for payload, p, r in zip(payloads, probs, reasons):
        p = float(p)
        results.append(
            {
                "churn_probability": round(p, 4),
                "risk_band": risk_band(p),
                "will_churn": p >= config.DECISION_THRESHOLD,
                "threshold": config.DECISION_THRESHOLD,
                "top_reasons": r,
                "retention_recommendation": recommend(payload, p),
                "model_version": _version,
            }
        )
    return results
