# Churn Platform

An **open-source churn prediction platform** you point at your own data.

Upload your customer CSV. The system profiles it, works out what the columns
mean, cleans and preprocesses them, and trains a model on *your* historical
churn. Every customer comes back with a churn score, a SHAP explanation of why,
and a suggested retention action — with tracking, deployment, monitoring, and
retraining around it.

Not a model trained on one public dataset. A system that trains a model on
yours.

> **Status: early.** A complete single-dataset implementation works end to end
> today — training, registry, API, SHAP, UI, Docker, CI. The multi-dataset
> platform work is in progress. See [ROADMAP.md](ROADMAP.md) for exactly what is
> built and what is not. Nothing here is oversold on purpose.

---

## Why another churn tool

Most churn projects stop at a probability. This one is built around three things
they usually skip:

**The cutoff is an economic decision, not 0.5.** Training sweeps the threshold
on out-of-fold predictions and picks the one that maximises expected profit,
given what a retention offer costs you and what a customer is worth. A model
that flags 26% of your base at 0.5 is not answering the question a business
asked.

**The threshold travels with the model.** It is chosen during training, logged
against that run, and read back by the API at load time. It cannot drift out of
sync with the model serving it. The schema is being moved onto the same footing.

**The explanation is per customer, and it is honest about what it is.** SHAP
tells you what the *model* used, not what *causes* churn. The roadmap treats
that as a problem to solve with a measured holdout group, not a caveat to bury.

---

## The core invariant

One model version carries three things that must always travel together:

```
model version = the fitted Pipeline
              + the schema it was trained on
              + the decision threshold chosen for it
```

If those ever separate, predictions are silently wrong — right-looking numbers,
wrong cutoff, or features in the wrong columns. Every design decision in this
repo protects that invariant.

The API loads **one object**: an sklearn `Pipeline` whose first step is the
preprocessor and whose last step is the classifier, accepting a raw `DataFrame`.
Never a bare booster with the encoders left behind in a notebook — that
guarantees train/serve skew.

---

## Quickstart

```bash
make install
# place a churn CSV at data/telco_churn.csv
make train        # trains, logs to MLflow, registers the model
make mlflow       # UI at :5000 to compare runs
make api          # API docs at :8000/docs
make install-ui   # Streamlit deps, kept out of the API image
make ui           # portal at :8501 (needs the API running)
make test
make up           # mlflow + api + ui together via compose
```

Fastest way to see it work: `make api`, then open **http://localhost:8000/docs**,
expand `POST /predict`, click **Try it out** → **Execute**.

> **Known gap:** `make up` starts a fresh MLflow with an empty database, so the
> containerised API has no model to load and reports `degraded`. Local runs are
> unaffected. Fixed in Stage 4 of the roadmap, when training becomes a service.

---

## Endpoints

| Route | Purpose |
| --- | --- |
| `GET /health` | Liveness, model version, active threshold and where it came from |
| `POST /predict` | One customer → probability, risk band, top SHAP reasons, retention hint |
| `POST /predict/batch` | Up to 1000 customers; `explain=false` by default for speed |
| `POST /reload` | Pull the current Production model without restarting the container |

`/health` returns `degraded` rather than a 500 when no model is loaded, so a
missing model is visible instead of looking like a crash.

`/reload` is what makes the registry meaningful — promote a new version in
MLflow, call `/reload`, done. No rebuild, no downtime.

Watch the `threshold_source` field in responses. If it ever reads `fallback`,
something upstream failed to log a threshold and the cutoff is an arbitrary 0.5.
That is deliberately visible rather than silent.

---

## Architecture

```
CSV upload ──► profile & map columns ──► quality + leakage checks
                                              │
                                              ▼
                                    train (XGBoost + baseline)
                                    pick threshold on cost
                                              │
                                              ▼
                            MLflow registry: pipeline + schema + threshold
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                       FastAPI /predict                 monitoring
                       score + SHAP + action            drift · decay · retrain
                              │
                              ▼
                       Streamlit portal
                    (HTTP only — never loads a model)
```

The UI talks to the API over HTTP and never imports a model. A second in-process
copy could drift from the served one, which is the exact failure the registry
and `/reload` design exists to prevent.

---

## Honest caveats

- **SHAP reasons are correlational.** "High monthly charges drove this
  prediction" is not "a discount will retain this customer." Retention
  recommendations are hypotheses until measured against a holdout group.
- **A high churn score is not the same as being worth saving.** Some customers
  will leave regardless, and spending on them is wasted. Uplift modelling is the
  real answer; this platform does not pretend to do it yet.
- **Automatic column mapping will sometimes be wrong.** What was inferred is
  always shown for correction before training.
- **Auto-training on unfamiliar data is risky without leakage checks.** A column
  recorded after churn produces a near-perfect model that is worthless in
  production. Detecting that is Stage 2 of the roadmap, and it is not done yet.

---

## Layout

```
src/
  schema.py       feature contract (Telco today; becoming a learned artifact)
  labels.py       readable names for encoded SHAP feature columns
  config.py       env-driven settings
  data.py         cleaning applied identically at train and serve time
  preprocess.py   ColumnTransformer built from the schema
  train.py        MLflow-tracked training + cost-based threshold selection
  api/
    models.py     pydantic request/response validation
    predict.py    model loading, inference, SHAP, retention hook
    main.py       FastAPI app
app/
  streamlit_app.py  portal; talks to the API over HTTP, never loads a model
tests/            run green with no model present
.github/workflows/ci.yml
```

---

## Deployment

Render or Railway take the Dockerfile directly. Set `MLFLOW_TRACKING_URI`,
`MODEL_NAME`, and `MODEL_STAGE`. Leave `DECISION_THRESHOLD` unset so the model's
own threshold is used — set it only as a deliberate operator override.

Tighten the CORS `allow_origins` list before exposing anything publicly.

Churn data is PII-heavy. The intended deployment story is self-hosted, with data
never leaving your own infrastructure.

---

## Contributing

Early days — the roadmap is the plan of record and the stages are ordered by
dependency. Stage 1 (moving the schema out of source code and into a per-dataset
artifact) unblocks everything else.

Commit messages are plain sentences describing what the change does. No `type:`
prefixes.
