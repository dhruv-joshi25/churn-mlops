# Customer Churn Prediction — MLOps Service

Serving and infrastructure half of the churn project. Sanjida owns modelling,
SHAP, and retention logic; this repo owns the tracking, API, container, and CI
that her model plugs into.

---

## The contract (read this first)

The API loads **one object**: an sklearn `Pipeline` whose first step is the
preprocessor and whose last step is the classifier. It must accept a raw
`DataFrame` with exactly the columns in `src/schema.py` — no pre-encoded input.

```python
import mlflow
mlflow.sklearn.log_model(pipe, "model", registered_model_name="churn-classifier")
```

Do **not** log a bare XGBoost booster. If the encoders live in the notebook and
only the booster is saved, the API has to reimplement preprocessing by hand, and
train/serve skew is guaranteed. One `Pipeline`, one artifact.

Two rules that follow from this:

1. `src/preprocess.py` pins OneHotEncoder categories from `schema.py` instead of
   inferring them. An unseen category at serve time then cannot shift column
   positions silently.
2. The pipeline is fit *inside* CV folds (`cross_val_predict` in `train.py`),
   so the scaler and imputer never see validation data.

---

## Division of work

| Area | Owner |
| --- | --- |
| EDA, feature engineering, model selection, SHAP, retention rules | Sanjida |
| `schema.py`, `preprocess.py`, MLflow instrumentation | Shared — agree before changing |
| FastAPI service, Docker, compose, CI, monitoring | Dhruv |

Sanjida's retention rules drop into `recommend()` in `src/api/predict.py`
without touching any endpoint code.

---

## Quickstart

```bash
make install
# put the Telco CSV at data/telco_churn.csv
make train        # logs to MLflow, registers the model
make mlflow       # UI at :5000 to compare runs
make api          # docs at :8000/docs
make install-ui   # Streamlit deps, kept out of the API image
make ui           # portal at :8501 (needs the API running)
make test
make up           # mlflow + api + ui together via compose
```

## Endpoints

| Route | Purpose |
| --- | --- |
| `GET /health` | Liveness + whether a model is loaded. Returns `degraded`, not 500, when no model exists. |
| `POST /predict` | One customer → probability, risk band, top SHAP reasons, retention hint |
| `POST /predict/batch` | Up to 1000 customers; `explain=false` by default for speed |
| `POST /reload` | Pull the current Production model without restarting the container |

`/reload` is what makes the registry meaningful — promote a new version in
MLflow, hit `/reload`, done. No rebuild.

---

## Threshold selection

`train.py` does not use 0.5. It sweeps the cutoff on **out-of-fold training
predictions** and picks the one maximising expected profit, given two numbers at
the top of the file:

```python
COST_RETENTION_OFFER = 200.0    # cost of a discount to one customer
VALUE_CUSTOMER_SAVED = 1200.0   # value of retaining a would-be churner
```

Set these with the team and state the assumption in your report. Churn is ~26%
positive, so accuracy is meaningless and F1 is arbitrary; expected value is the
metric a business person can actually argue with. This is usually the single
strongest thing in a churn project presentation.

Both PR-AUC and ROC-AUC are logged. On imbalanced data, prefer PR-AUC when
comparing runs.

---

## Honest caveats to state in the report

- **SHAP reasons are correlational.** "High monthly charges drove this
  prediction" is not "a discount will retain this customer." Retention
  recommendations are hypotheses; validating them needs an A/B test or uplift
  modelling. Say this explicitly rather than letting a reader assume causality.
- **The dataset is static.** There is no genuine drift to monitor on 7043 fixed
  rows. If monitoring must be demonstrated, simulate batches over time (e.g.
  slice by `tenure`) and show a real degrade → retrain → promote loop rather
  than claiming production monitoring that isn't happening.
- `TotalCharges` ≈ `tenure × MonthlyCharges`. Keep it or drop it deliberately,
  and be ready to justify the choice.

---

## Layout

```
src/
  schema.py       feature contract, shared by training and serving
  labels.py       readable names for encoded SHAP feature columns
  config.py       env-driven settings
  data.py         cleaning applied identically at train and serve time
  preprocess.py   ColumnTransformer with pinned categories
  train.py        MLflow-tracked training + threshold selection
  api/
    models.py     pydantic request/response validation
    predict.py    model loading, inference, SHAP, retention hook
    main.py       FastAPI app
app/
  streamlit_app.py  portal; talks to the API over HTTP, never loads a model
tests/            run green with no model present
.github/workflows/ci.yml
```

## Deployment

Render or Railway take the Dockerfile directly. Set `MLFLOW_TRACKING_URI`,
`MODEL_NAME`, `MODEL_STAGE`, and `DECISION_THRESHOLD` as environment variables.
Tighten the CORS `allow_origins` list before exposing anything publicly.
