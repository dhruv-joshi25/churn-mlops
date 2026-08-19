# Customer Churn Prediction — Build Roadmap

**Team:** Sanjida (ML, SHAP, retention logic) · Dhruv (MLOps, API, deployment)
**Status as of 19 Aug 2026:** XGBoost baseline trained. MLOps not started.

---

## How to use this document

Work through phases in order **within your own track**. The two tracks run in
parallel and meet at three **integration gates**. Do not start a phase until its
entry condition is met, and do not call a phase done until every box in its
Definition of Done is ticked.

Each phase says who owns it. If a phase says "both," it needs a call, not a
WhatsApp message.

---

## Phase 0 — Already done (LOCKED, do not redo)

This work is complete and correct. It is recorded here so nobody repeats it and
so the report has an accurate account of what was built.

| # | Item | Detail |
| --- | --- | --- |
| 0.1 | Dataset acquired | Telco Customer Churn — 7043 rows, 21 columns |
| 0.2 | Missing values checked | Blank strings identified in `TotalCharges` |
| 0.3 | `TotalCharges` converted | String → numeric |
| 0.4 | `customerID` dropped | Non-predictive identifier removed |
| 0.5 | EDA completed | `Contract`, `MonthlyCharges`, `tenure` analysed against churn |
| 0.6 | Categorical encoding | Categoricals encoded for modelling |
| 0.7 | Train/test split | 80/20 |
| 0.8 | XGBoost baseline | Trained successfully |

**One thing to verify, not redo (5 minutes):** confirm the encoder in 0.6 was
fit *after* the split in 0.7, not before. If it was fit on the full dataset, the
baseline metrics are mildly optimistic. Either way the model stands — just note
it, and the pipeline in Phase 2 fixes it going forward.

---

## The integration contract

Everything downstream depends on one agreement. Fix it now, in writing.

**The deliverable from the ML track is a single sklearn `Pipeline`** whose first
step is preprocessing and last step is the classifier. It accepts a raw
DataFrame with the 19 columns listed in `src/schema.py` — not pre-encoded input.

```python
import mlflow
mlflow.sklearn.log_model(pipe, "model", registered_model_name="churn-classifier")
```

Why this matters: if only the XGBoost booster is saved, the encoders stay in the
notebook as loose variables and the API has to reimplement preprocessing by
hand. Doing that identically is very hard, and getting it subtly wrong is the
single most common bug in deployed ML — the model sees different features in
production than it saw in training, and accuracy silently collapses.

**Column names are owned by `src/schema.py`.** Neither person renames a feature
without changing that file and telling the other.

---

## Work split

| Track A — Sanjida | Track B — Dhruv |
| --- | --- |
| Model evaluation & tuning | Repo, branches, CI |
| Pipeline packaging | MLflow server & registry |
| SHAP explainability | FastAPI service |
| Retention rule design | Docker & compose |
| Streamlit UI content | Deployment & monitoring |
| Report: modelling sections | Report: architecture sections |

---

# TRACK A — Machine Learning (Sanjida)

## Phase A1 — Evaluate the baseline properly

**Entry:** Phase 0 complete. ✅ You are here.

**Tasks**
1. Compute on the test set: precision, recall, F1, ROC-AUC, **PR-AUC**, and the
   confusion matrix.
2. Note the churn rate (~26.5%). Write down explicitly why accuracy is not being
   reported as a headline number.
3. Plot the precision-recall curve, not just ROC. On imbalanced data ROC-AUC
   looks flattering; PR-AUC is the honest comparison metric.

**Definition of Done**
- [ ] All six metrics recorded for the baseline
- [ ] Confusion matrix saved as an image for the report
- [ ] One paragraph written on why PR-AUC is the primary comparison metric

**Common failure:** reporting 80% accuracy as a success. A model predicting "no
churn" for everyone scores 73.5%. Always state that number as the baseline to
beat.

---

## Phase A2 — Handle imbalance and tune

**Entry:** A1 done.

**Tasks**
1. Set `scale_pos_weight = n_negative / n_positive` in XGBoost (≈2.77). Compare
   against `class_weight='balanced'` alternatives.
2. If trying SMOTE, apply it **inside CV folds only**, never before splitting.
   Honestly, `scale_pos_weight` usually wins on this dataset and is easier to
   defend — try SMOTE only if you have time.
3. Tune with 5-fold stratified CV: `max_depth` (3–6), `learning_rate`
   (0.01–0.1), `n_estimators`, `subsample`, `colsample_bytree`, `reg_lambda`.
4. Train Logistic Regression and Random Forest as comparison models. You need
   more than one model in the report, and logistic regression is the
   interpretable baseline that makes XGBoost's gain meaningful.

**Definition of Done**
- [ ] Three model families trained and compared on PR-AUC
- [ ] Best hyperparameters recorded
- [ ] Comparison table ready for the report

**Common failure:** tuning against the test set. Tune on CV over training data;
touch the test set once, at the end.

---

## Phase A3 — Choose the decision threshold

**Entry:** A2 done. **This is the most valuable phase in the project.**

A model outputs a probability. Turning it into a "will churn" decision needs a
cutoff, and 0.5 is an arbitrary default that nobody defends when asked.

**Tasks**
1. Agree two numbers with the team:
   - Cost of giving one retention offer (e.g. ₹200 discount)
   - Value of retaining one customer who would have churned (e.g. ₹1200 LTV)
2. Sweep thresholds 0.05→0.95 on **out-of-fold training predictions**.
3. At each threshold compute expected value:
   `TP × value_saved − (TP + FP) × cost_of_offer`
4. Pick the maximum. Record it. `src/train.py` already implements this —
   `pick_threshold()`.

**Definition of Done**
- [ ] Cost assumptions written down and justified
- [ ] Threshold chosen on out-of-fold data, not test data
- [ ] Expected-value curve plotted for the report
- [ ] Threshold communicated to Dhruv (it becomes `DECISION_THRESHOLD`)

**Why it matters:** this converts "our ROC-AUC is 0.84" into "this model saves
approximately ₹X per 1000 customers." That sentence is what an evaluator or an
interviewer actually remembers.

---

## Phase A4 — Package as a Pipeline

**Entry:** A3 done. **→ Feeds Integration Gate 1.**

**Tasks**
1. Rebuild the winning model as `Pipeline([('preprocess', ...), ('model', ...)])`
   using `src/preprocess.py`. Do not hand-encode.
2. Confirm `pipe.predict_proba(raw_df)` works on a raw, unencoded DataFrame.
3. Verify the refit pipeline reproduces A2's metrics (small differences are fine;
   large ones mean the encoding changed).

**Definition of Done**
- [ ] Pipeline accepts raw DataFrame
- [ ] Metrics reproduced within tolerance
- [ ] `pipe` handed to Dhruv or logged to MLflow

---

## Phase A5 — SHAP explainability

**Entry:** A4 done.

**Tasks**
1. `shap.TreeExplainer` on the model step, applied to preprocessed features.
2. Global: summary/beeswarm plot — which features drive churn overall.
3. Local: per-customer top-4 contributing features with direction.
4. Map one-hot column names back to readable labels (`Contract_Month-to-month`
   → "Month-to-month contract"). Nobody wants raw column names in a UI.

**Definition of Done**
- [ ] Global summary plot saved for the report
- [ ] Local explanation works for any single customer
- [ ] Readable label mapping written
- [ ] Sanity check: high-risk customers should surface month-to-month contract,
      fiber optic, electronic check, low tenure. If they don't, something is wrong.

---

## Phase A6 — Retention recommendation logic

**Entry:** A5 done.

**Tasks**
1. Write rules mapping risk drivers → offers. Start simple:
   - Month-to-month → discounted annual contract
   - No tech support + has internet → free support trial
   - High monthly charges → plan right-sizing
   - Electronic check → autopay incentive
2. Hand the rules to Dhruv; they drop into `recommend()` in
   `src/api/predict.py` without touching endpoint code.

**Definition of Done**
- [ ] Rules cover every high-risk driver from A5
- [ ] Each rule has a stated rationale
- [ ] **Causality caveat written for the report** (see below)

**Non-negotiable caveat:** SHAP explains what the *model* used, not what *causes*
churn. "High charges drove this prediction" ≠ "a discount will retain them."
Recommendations are hypotheses requiring an A/B test to validate. Write this
sentence in the report. Claiming causal effect from SHAP is the mistake an
evaluator will catch, and pre-empting it turns a weakness into a strength.

---

# TRACK B — MLOps (Dhruv)

## Phase B1 — Repository and CI

**Entry:** Immediate. Nothing blocks this.

**Tasks**
1. Push the scaffold to GitHub. Branches: `main` (protected), `develop`, feature
   branches off `develop`.
2. Enable the CI workflow (`.github/workflows/ci.yml`). Require it to pass before
   merge to `main`.
3. Add both collaborators. Sanjida commits notebooks to `notebooks/`.
4. Confirm `make test` is green — it passes with no model present by design.

**Definition of Done**
- [ ] Repo live, both members have access
- [ ] CI green on `develop`
- [ ] `main` protected, requires passing CI
- [ ] README's contract section read by both people

---

## Phase B2 — MLflow tracking

**Entry:** B1 done.

**Tasks**
1. Run the MLflow server (`docker compose up mlflow`, or `make mlflow` for a
   local file store).
2. Run `python -m src.train --model logistic` end to end even with a weak model —
   the point is proving the round trip works, not the score.
3. Confirm params, metrics, and the model artifact all appear in the UI.
4. Show Sanjida the UI so her tuning runs get logged and become comparable.

**Definition of Done**
- [ ] MLflow reachable, experiments visible
- [ ] At least one run logged with a model artifact
- [ ] Sanjida can log runs from her notebook

---

## 🚦 INTEGRATION GATE 1 — Round trip proven

**Both people. Do this as early as possible, with whatever model exists.**

The purpose is to prove the plumbing works. A bad model proves it as well as a
good one, so do **not** wait for the final tuned model.

1. Sanjida logs any pipeline to MLflow
2. Register it and promote to `Production` stage
3. Dhruv starts the API, calls `POST /predict` with the example payload
4. A probability comes back

**Gate passed when:** a raw JSON customer produces a churn probability.

If this is left until week 4, integration bugs surface at the worst possible
time. Do it in week 1.

---

## Phase B3 — FastAPI service

**Entry:** Gate 1 passed.

**Tasks**
1. Verify all four endpoints: `/health`, `/predict`, `/predict/batch`, `/reload`.
2. Set `DECISION_THRESHOLD` to the value from A3.
3. Test rejection cases — invalid category, negative tenure — return 422.
4. Confirm `/health` returns `degraded` rather than crashing when no model loads.

**Definition of Done**
- [ ] `/docs` renders interactive Swagger UI
- [ ] Batch endpoint handles 100+ customers
- [ ] `/reload` picks up a newly promoted model without restart
- [ ] Tests pass

---

## Phase B4 — Wire in SHAP and retention

**Entry:** B3 done, A5 and A6 delivered.

**Tasks**
1. Confirm `/predict?explain=true` returns top reasons.
2. Replace the placeholder rules in `recommend()` with Sanjida's actual rules.
3. Apply the readable-label mapping from A5 to SHAP feature names.
4. Measure latency with explanations on. If slow, keep `explain=false` as the
   batch default (already the case).

**Definition of Done**
- [ ] Single prediction returns probability + band + reasons + recommendation
- [ ] Feature names are human-readable
- [ ] Single prediction under ~500ms

---

## Phase B5 — Docker

**Entry:** B4 done.

**Tasks**
1. `make docker` — build succeeds.
2. Run the container, hit `/health` from outside.
3. `docker compose up` — API and MLflow talk to each other.
4. Confirm the CI docker job passes.

**Definition of Done**
- [ ] Image builds clean
- [ ] Container healthcheck passes
- [ ] Compose stack works end to end
- [ ] Image size noted for the report

---

## Phase B6 — Streamlit portal

**Entry:** B5 done. **Both people — Sanjida owns content, Dhruv owns wiring.**

**Tasks**
1. `app/streamlit_app.py` calling the API over HTTP — never importing the model
   directly. The UI must not have its own copy of the model; that defeats the
   whole architecture.
2. Two modes: single-customer form, and CSV upload for batch.
3. Display: probability gauge, risk band, top reasons, recommendation.
4. Add a `Dockerfile.streamlit` and uncomment the `ui` service in
   `docker-compose.yml`.

**Definition of Done**
- [ ] Single prediction works from the browser
- [ ] CSV upload returns a scored table
- [ ] UI reads `API_URL` from environment, no hardcoded localhost
- [ ] Whole stack runs with one `docker compose up`

---

## 🚦 INTEGRATION GATE 2 — Full stack local

**Both people.** One command brings up MLflow + API + Streamlit. A customer
entered in the browser returns probability, reasons, and a recommendation.

**Take a screen recording here.** This is the demo. Do not rely on live demos
working on presentation day.

---

## Phase B7 — Deployment

**Entry:** Gate 2 passed.

**Tasks**
1. Deploy the API container to Render (free tier takes the Dockerfile directly).
2. Set env vars: `MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_STAGE`,
   `DECISION_THRESHOLD`.
3. Deploy Streamlit to Streamlit Community Cloud, pointed at the API URL.
4. **Tighten CORS** — replace `allow_origins=["*"]` with the Streamlit domain.
5. Expect a cold-start delay on free tiers; mention it rather than being
   surprised by it in the demo.

**Definition of Done**
- [ ] Public API URL responding to `/health`
- [ ] Public Streamlit URL producing predictions
- [ ] CORS restricted
- [ ] Both URLs in the README

---

## Phase B8 — Monitoring (scope this honestly)

**Entry:** B7 done.

The roadmap image says "monitor model performance." On 7043 static rows there is
no genuine drift to detect. Pick one of two honest options:

**Option 1 — Service monitoring (simpler, fully truthful)**
- Request count, latency percentiles, error rate
- Prediction distribution over time — if the mean predicted probability shifts,
  the input population has changed
- `/health` polled by an uptime checker

**Option 2 — Simulated drift (more impressive, more work)**
- Slice the dataset into pseudo-monthly batches by `tenure`
- Score each batch, track PR-AUC across batches
- Introduce deliberate drift in one batch, detect it, retrain, promote a new
  version in the registry, call `/reload`
- Demonstrate the full degrade → retrain → promote → serve cycle

**Definition of Done**
- [ ] One option implemented fully
- [ ] Report states plainly which one and why the other was out of scope

**Do not claim production drift monitoring you are not doing.** Overclaiming is
what turns a good project into a challenged one.

---

## 🚦 INTEGRATION GATE 3 — Submission ready

- [ ] Public URLs live
- [ ] README complete with architecture diagram
- [ ] Report written, both sections merged
- [ ] Demo recording saved
- [ ] Repo clean — no notebooks with 200 output cells, no committed CSV, no keys
- [ ] Both members can explain the *other* person's half

That last box matters. If only one person can explain the deployment, the
project reads as one person's work.

---

## Suggested sequencing

| Week | Sanjida | Dhruv | Milestone |
| --- | --- | --- | --- |
| 1 | A1, A2 | B1, B2 | **Gate 1** — round trip |
| 2 | A3, A4 | B3 | Threshold fixed, API live |
| 3 | A5, A6 | B4, B5 | Explanations + container |
| 4 | UI content | B6, B7 | **Gate 2** — full stack |
| 5 | Report | B8, docs | **Gate 3** — submission |

Compress if the deadline is tighter, but never move Gate 1 later. Everything
else can be rushed; unproven integration cannot.

---

## Risk register

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Bare booster saved instead of Pipeline | High | Contract agreed now; Gate 1 catches it |
| Threshold mismatch between notebook and API | High | Single value in `DECISION_THRESHOLD`, set in A3 |
| Train/serve encoding skew | Medium | Pinned categories in `schema.py` |
| Streamlit loads model directly, bypassing API | Medium | Code review at B6 |
| Free-tier cold start ruins live demo | Medium | Pre-recorded demo |
| SHAP too slow on batch | Low | `explain=false` default already set |
| Integration left to final week | High | Gate 1 in week 1 |

---

## Report and viva talking points

Prepare answers to these. They are the questions that actually get asked.

1. **"Why not accuracy?"** — 26.5% positive class; predict-all-negative scores
   73.5%. PR-AUC and expected value instead.
2. **"Why 0.5 threshold?"** — It isn't. Cost-based selection on out-of-fold
   predictions, with stated cost assumptions. *This is your strongest answer.*
3. **"Does SHAP prove causation?"** — No. Correlational attribution of model
   behaviour. Recommendations are hypotheses needing an A/B test.
4. **"Why XGBoost over logistic regression?"** — Show the comparison table from
   A2. If the gain is small, say so; defensible honesty beats an inflated claim.
5. **"What's actually MLOps here?"** — Experiment tracking, model registry with
   stage promotion, containerised serving, CI on every push, hot model reload
   without redeploy.
6. **"How would this work at real scale?"** — Feature store for consistency,
   scheduled retraining on fresh data, uplift modelling to replace rule-based
   retention, monitoring on genuine incoming traffic.
7. **"What would you do differently?"** — Uplift modelling instead of
   classification-plus-rules; a dataset with real timestamps enabling true
   temporal validation.

---

## Git conventions

```
main      protected, always deployable
develop   integration branch
feat/...  features off develop
```

Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

Do not commit: the CSV, `mlruns/`, `.env`, model binaries. All already in
`.gitignore`.

---

## Weekly sync template

Fifteen minutes, once a week, both people:

1. What phase am I on, and is its Definition of Done met?
2. Anything blocking the other person?
3. Has anything changed in `schema.py`, the threshold, or the contract?
4. Is the next gate still on track?

Question 3 is the one that prevents the expensive bugs.
