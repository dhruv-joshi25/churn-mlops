import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{ROOT / 'mlruns'}")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "churn-prediction")
REGISTERED_MODEL_NAME = os.getenv("MODEL_NAME", "churn-classifier")

# Which registry stage the API serves. "Production" once a model is promoted.
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")

# Fallback for local dev / CI when no registry is reachable.
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", str(ROOT / "models" / "model"))

DATA_PATH = os.getenv("DATA_PATH", str(ROOT / "data" / "telco_churn.csv"))

# Decision threshold. NOT 0.5 by default — see README on cost-based selection.
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
