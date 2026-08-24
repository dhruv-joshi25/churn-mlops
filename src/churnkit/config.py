import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent

# Where relative defaults (mlruns/, models/, data/) hang off.
#
# In a source checkout the package sits at <repo>/src/churnkit, so the repo root
# is two levels up and the defaults point at the developer's own directories no
# matter which directory they run from. Once churnkit is installed as a package
# that reasoning stops holding — site-packages has no data/ — so the working
# directory takes over, which is what the containers rely on: they install the
# package and run from /app, where models/ and data/ are mounted.
ROOT = (
    _PACKAGE_DIR.parents[1]
    if _PACKAGE_DIR.parent.name == "src"
    else Path.cwd()
)

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

# Platform code names no dataset (I11), so the default is a generic filename and
# the real path comes from the environment. .env.example carries the path the
# reference implementation trains on.
DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "data" / "customers.csv"))

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

# ── Browser-facing hardening ──────────────────────────────────────────────────
#
# None of this is user authentication, and none of it is meant to become that
# (ADR 0001: one operator, one deployment, no accounts). It is a same-origin
# boundary, which is a different thing: it stops a page the operator happens to
# have open in another tab from driving their API on their behalf.

# Origins allowed to call the API from a browser. The portal is not in this list
# because it needs to be — Streamlit calls the API server-side, so no browser
# request crosses an origin — but because an operator writing their own page
# against a local deployment expects it to work. See ADR 0004.
_raw_origins = os.getenv("CORS_ORIGINS", "").strip()
CORS_ORIGINS = (
    [origin.strip() for origin in _raw_origins.split(",") if origin.strip()]
    if _raw_origins
    else ["http://localhost:8501", "http://127.0.0.1:8501"]
)

# Header every state-changing request must carry. The value is irrelevant and is
# not a secret: what matters is that it is a *custom* header, because a browser
# will not send one cross-origin without a preflight, and CORS_ORIGINS is what
# refuses that preflight. The two together are the CSRF defence; neither half
# works alone.
ADMIN_HEADER = "X-Churnkit-Admin"
