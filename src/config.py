import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load .env for local development. Optional by design: Docker and CI inject
# environment variables directly, so the app must work without this file.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT / 'mlruns'}")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "churn-prediction")
REGISTERED_MODEL_NAME = os.getenv("MODEL_NAME", "churn-classifier")

# Which registry stage the API serves. "Production" once a model is promoted.
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Fallback for local dev / CI when no registry is reachable.
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", str(ROOT / "models" / "model"))

DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "data" / "telco_churn.csv"))

# Decision threshold.
#
# The authoritative value is the one train.py chose by maximising expected value
# on out-of-fold predictions; it is logged as the `decision_threshold` tag on the
# run that produced the model, and the API reads it back at load time. That way
# the cutoff travels with the model version and cannot drift out of sync.
#
# Setting DECISION_THRESHOLD in the environment is an explicit operator override
# and wins over the model's own value. Leaving it unset — the normal case — lets
# the model speak for itself. An unset variable is None here, not 0.5, so that a
# missing value can never masquerade as a deliberate choice.
_raw_threshold = os.getenv("DECISION_THRESHOLD", "").strip()
DECISION_THRESHOLD_OVERRIDE = float(_raw_threshold) if _raw_threshold else None

# Last-resort fallback, used only when the model carries no threshold and no
# override is set. 0.5 is not a defensible choice — if the API reports this
# value, something upstream failed to log the threshold.
FALLBACK_THRESHOLD = 0.5

# Backwards-compatible alias for anything still importing the old name.
DECISION_THRESHOLD = (
    DECISION_THRESHOLD_OVERRIDE
    if DECISION_THRESHOLD_OVERRIDE is not None
    else FALLBACK_THRESHOLD
)
