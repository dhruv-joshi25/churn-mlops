# Churn Platform — Build Roadmap

**Team:** Sanjida (ML, SHAP, retention logic) · Dhruv (platform, API, MLOps)
**Status as of 20 Aug 2026:** Telco reference implementation working end to end.
Platform pivot starting.

---

## What we are actually building

An **open-source churn prediction platform** that any company can point at its
own data. Not a model trained on one fixed dataset — a system that trains a
model on *your* data:

1. A company uploads its customer CSV.
2. The system profiles it — identifies and maps the columns, works out what is
   numeric, categorical, a date, an identifier, and which column is the target.
3. It cleans and preprocesses accordingly.
4. It trains XGBoost on that company's historical churn.
5. Every customer gets a churn score, a SHAP explanation of *why*, and a
   suggested retention action.
6. MLOps around all of it: tracking, deployment, monitoring, and retraining as
   new data arrives.

The test of every decision in this document: **would it still work if the next
CSV had completely different columns?** If the answer is no, it is not
platform code.

---

## What exists today, honestly

The repository currently contains a **working single-dataset implementation**
built on the Telco churn CSV. It is not wasted work — it proves the serving
architecture and it is the reference every platform component gets checked
against. But it hardcodes one schema.

| Built and verified | Where |
| --- | --- |
| Training with MLflow tracking + model registry | `src/train.py` |
| Cost-based threshold, logged with the model | `src/train.py`, `src/api/predict.py` |
| FastAPI service — predict, batch, health, reload | `src/api/` |
| SHAP reasons with readable labels | `src/api/predict.py`, `src/labels.py` |
| Streamlit portal, talks to the API over HTTP only | `app/streamlit_app.py` |
| Docker, compose, CI, tests | root, `.github/workflows/` |

Trained model in the registry: `churn-classifier` v2, Production,
PR-AUC 0.658, recall 0.87, threshold 0.31.

**Known gap:** the compose stack starts a fresh MLflow with an empty database,
so the containerised API has no model to load and reports `degraded`. Fixed in
Stage 4 when training becomes a service rather than a local script.

---

## The one change everything depends on

Right now the feature contract lives in **source code**. `src/schema.py` lists
Telco's 19 columns and pins every allowed category value, and the rest of the
system reads it at import time:

- `src/preprocess.py:28` — OneHotEncoder categories come from `CATEGORY_VALUES`
- `src/data.py:33` — the target is literally compared against `"Yes"`
- `src/api/models.py` — every request field is a `Literal[...]` fixed at import
- `app/streamlit_app.py` — form dropdowns are rendered from the same lists

Pinning the schema was the *right* decision for one dataset: it is what stops
train/serve skew. But a platform learns its schema at upload time, so the schema
has to stop being code and start being **data** — inferred from the CSV, saved
as an artifact next to the model, and loaded back at serve time.

We have already solved this exact problem once. The decision threshold is chosen
during training, logged with the model, and read back by the API so it can never
drift out of sync. **The schema gets the same treatment.** One model version
carries three things that must always travel together:

```
model version = the fitted Pipeline
              + the schema artifact it was trained on
              + the decision threshold chosen for it
```

If those three ever separate, predictions are silently wrong. That is the single
invariant this whole platform rests on.

---

## Stage 1 — Schema becomes data

**Owner:** Dhruv. **Blocks everything else.**

**Tasks**
1. Write `src/profiler.py` — reads a DataFrame, returns a schema object: per
   column, its role (target / id / date / numeric / categorical / drop), dtype,
   observed categories, missing rate, cardinality.
2. Serialise that schema to JSON and log it as an MLflow artifact on the run.
3. Rewrite `build_preprocessor()` to take a schema argument instead of importing
   pinned constants.
4. Rewrite `clean()` and `split_xy()` to be schema-driven — no column named in
   code.
5. Build the pydantic request model at runtime from the loaded schema
   (`pydantic.create_model`) instead of declaring it statically.
6. Render the Streamlit form from the loaded schema.
7. Keep `schema.py` as a Telco *fixture* used by tests, not as the live contract.

**Definition of Done**
- [ ] A CSV with entirely different column names trains without code changes
- [ ] The API validates requests against the uploaded schema, not Telco's
- [ ] Telco still trains and serves with identical metrics to today
- [ ] Schema artifact visible on the MLflow run

**Failure to avoid:** inferring categories at serve time instead of reading the
saved artifact. That reintroduces exactly the skew the pinning was preventing.

---

## 🚦 GATE 1 — Two different datasets, one codebase

Train on Telco and on any second, structurally different churn CSV. Both serve
predictions through the same API with no code edited between them.

Until this passes, nothing else is worth building.

---

## Stage 2 — Make it trustworthy

**Owner:** Sanjida leads the statistics, Dhruv wires them in.

This stage is what separates a real product from a demo. On arbitrary company
data, a model that looks excellent is usually broken.

**2.1 Leakage detection** — the highest-value item in this document.
A column like `cancellation_reason` or `refund_date` is recorded *after* churn
and will produce a near-perfect model that is worthless in production. Flag
columns that predict the target almost alone, columns whose missingness aligns
with the target, and columns named like post-outcome events. Report them to the
user and exclude by default:
> *Dropping `exit_survey_score` — it appears to be recorded after the customer
> left, so it cannot be used to predict churn in advance.*

**2.2 Churn label builder**
Most companies do not have a `Churn` column. They have subscriptions or
transactions, and churn must be *defined*: "no purchase in 90 days",
"subscription ended and not renewed". Without this, the platform only serves
companies that already did the hard part.

**2.3 Time-based validation**
A random train/test split leaks the future into the past and overstates churn
performance. When a date column exists, split temporally — train before date T,
test after.

**2.4 Probability calibration**
Raw XGBoost scores are not probabilities. If the UI says "72% likely to churn",
that should mean roughly 72 of 100 such customers actually leave. Add isotonic
or Platt calibration and log a reliability curve.

**2.5 Data-quality gate**
Row count, churner count, missingness, constant columns, duplicate IDs. Refuse
politely when the data cannot support a model rather than training anyway:
> *Only 41 churned customers found. Results will not be reliable below ~200.*

**2.6 Always train a baseline**
Run logistic regression alongside XGBoost and report both. If the simple model
wins on their data, say so.

**Definition of Done**
- [ ] A deliberately leaky column is caught and reported on a test CSV
- [ ] Churn can be derived from a dataset with no churn column
- [ ] Calibration curve logged per run
- [ ] Training refuses on data that is too small or too imbalanced

---

## Stage 3 — Make it actionable

**Owner:** Sanjida owns the business logic, Dhruv owns the plumbing.

A probability is not a decision. This stage turns scores into something a
retention team acts on.

**3.1 Per-company cost inputs**
The threshold logic is already the strongest idea in the project, but
`COST_RETENTION_OFFER = 200` and `VALUE_CUSTOMER_SAVED = 1200` are Telco
guesses. Ask each company what an offer costs them and what a customer is worth,
store it with their project, and derive *their* cutoff from *their* economics.

**3.2 Revenue at risk**
Rank by value × probability, not probability alone. "₹18.2L of ARR at risk"
moves a business; "347 customers flagged" does not.

**3.3 Action worklist export**
The real Monday-morning deliverable: top N customers with score, top reasons,
recommended offer, and the expected value of intervening — as CSV, later pushed
to a CRM. This is the product. The model is just how it gets computed.

**3.4 Retention playbooks**
Replace the hardcoded rules in `recommend()` with rules a company writes against
their own features and SHAP drivers, plus starter templates per industry.

**Definition of Done**
- [ ] Threshold derived from company-supplied costs
- [ ] Worklist CSV downloadable from the UI
- [ ] Zero Telco-specific strings left in `recommend()`

---

## Stage 4 — Make it a platform

**Owner:** Dhruv.

**4.1 Projects and isolation** — a company's uploads, schemas, models, and
settings live under a project id. One MLflow experiment and one registered model
name per project.

**4.2 Training as an async job** — upload → queue → profile → quality gate →
train → report. Not a CLI invocation. Status visible while it runs.

**4.3 Model cache per project** — `predict.py` currently holds one global
`_model` singleton. It needs a keyed cache with the schema and threshold bound
to each entry.

**4.4 Log every prediction** — features, score, model version, timestamp.
Cheap now, impossible to reconstruct later, and a hard prerequisite for
everything in Stage 5.

**4.5 Fix the compose gap** — with training as a service, the containerised
MLflow gets populated properly and `/health` stops reporting `degraded`.

**Definition of Done**
- [ ] Two projects with different schemas served by one API simultaneously
- [ ] Upload to scored output with no shell access
- [ ] `docker compose up` produces a working stack from empty state

---

## 🚦 GATE 2 — Upload to insight, in a browser

A person who has never seen the code uploads a CSV, waits, and gets scored
customers with reasons and recommended actions. No terminal involved.

**Record this.** It is the demo.

---

## Stage 5 — Monitoring, retraining, and proof

**Owner:** Dhruv, with Sanjida on the measurement design.

Unlike the old fixed-dataset project, a real platform genuinely receives new
data over time, so monitoring here is honest rather than simulated.

**5.1 Drift monitoring** — per-feature PSI against the training distribution,
plus prediction distribution shift. A population that has changed shape is worth
flagging even before performance drops.

**5.2 Delayed-label performance tracking** — once outcomes mature, compare
logged predictions against what actually happened. This is real model decay,
measured, not claimed.

**5.3 Champion/challenger retraining** — a retrained model must beat the
incumbent on a holdout before promotion. Auto-promote or require approval.

**5.4 Holdout control group** — the most valuable feature in this document.
Automatically hold back a random slice of flagged customers as *untreated*, then
compare their churn rate against the treated group. This turns SHAP-derived
recommendations from hypotheses into measured money, and it is the honest answer
to the causality problem below. Very few tools ship this.

**Definition of Done**
- [ ] Drift visible per feature over time
- [ ] A retrain triggered by measured decay, not a calendar
- [ ] Holdout results reported as an actual retention lift number

---

## Stage 6 — Open source properly

**Owner:** Dhruv.

1. One-command self-host: `docker compose up` brings up API, worker, MLflow,
   Postgres, object storage, and UI from empty state.
2. **Data never leaves the company's infrastructure.** Churn data is PII-heavy;
   this is a genuine adoption advantage and it should be stated plainly.
3. Identifiers hashed or dropped by default.
4. LICENSE, CONTRIBUTING, an architecture diagram, and a sample dataset that is
   not Telco.
5. Honest README — see the caveats section there.

---

## Honest caveats — keep these in the README, do not quietly drop them

- **SHAP explains the model, not the world.** "High monthly charges drove this
  prediction" is not "a discount will retain this customer." Recommendations are
  hypotheses. Stage 5.4 is how they stop being hypotheses.
- **A high churn score does not mean a customer is worth spending on.** Some
  will leave regardless; spending on them is wasted. Proper uplift modelling is
  the long-term answer, and until then the platform should not pretend otherwise.
- **Auto-training on unknown data is dangerous without Stage 2.** A confident
  model built on leaked columns is worse than no model, because someone will act
  on it.
- **Automatic column mapping will sometimes be wrong.** Always show the user
  what was inferred and let them correct it before training.

---

## Build order

| Order | Stage | Why here |
| --- | --- | --- |
| 1 | Stage 1 — schema as data | Nothing else is possible first |
| 2 | Stage 2 — trustworthy | Wrong models are worse than none |
| 3 | Stage 3 — actionable | Turns scores into a product |
| 4 | Stage 4 — platform | Multi-tenant, jobs, isolation |
| 5 | Stage 5 — monitoring | Needs logged predictions from Stage 4 |
| 6 | Stage 6 — open source | Package once the thing works |

Stages 2 and 3 can overlap. Stage 5 cannot start before 4.4 is logging.

---

## Git conventions

```
main      protected, always deployable
develop   integration branch
feat/...  features off develop
```

Commit messages: a plain sentence saying what the change does, capitalised, with
no `type:` prefix. Add body paragraphs when the reasoning is worth recording.

Do not commit: customer CSVs, `mlruns/`, `.env`, model binaries.

---

## Weekly sync

1. What stage am I on, and is its Definition of Done met?
2. Anything blocking the other person?
3. Has the schema artifact format, the threshold logic, or the model-version
   contract changed?

Question 3 is the one that prevents the expensive bugs.
