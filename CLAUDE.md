# CLAUDE.md — Project Constitution

**Read this file at the start of every session. Re-read before every commit.**

If any instruction conflicts with this file, this file wins. Stop and say so
rather than proceeding.

---

## What this is

An open-source churn intelligence **platform**. A company points it at their own
customer data and gets, without writing code:

1. **Upload** — a CSV (and optionally an event log) of their customers
2. **Automatic schema mapping** — the system identifies which column is the
   customer ID, which is the churn target, which is the timestamp, and what each
   remaining column is. It proposes a complete mapping with confidence scores.
3. **Cleaning and preprocessing** — encodings, delimiters, date formats, null
   spellings, currency-polluted numerics, all handled without silent drops
4. **Training on their own history** — XGBoost, fitted to their data, their
   churn definition, their time windows
5. **Per-customer churn score** — calibrated, so 0.72 means 72 in 100
6. **SHAP explanation** — which factors drove each customer's score
7. **Retention action** — a suggested intervention per at-risk customer
8. **The MLOps loop** — every run tracked, models registered and served,
   drift monitored, retraining as new data arrives

**This is not a model for one fixed dataset.** Every company that runs it gets a
model trained on their data with their column names. Nothing about the Telco
dataset may be hardcoded anywhere in `src/`.

## Deployment model — self-hosted, single-tenant

One operator, one deployment, one dataset at a time. The company runs the
container on their own infrastructure. **Their data never leaves their machine.**

Not multi-tenant. Not hosted by us. No auth, no user accounts, no job queue.
Do not build toward multi-tenancy. Do not add `tenant_id` anywhere. If a task
seems to need tenancy, it is out of scope — stop and ask. See ADR 0001.

"Any company can upload their CSV" is satisfied by the upload → auto-map →
train flow inside one deployment. It does not require accounts.

## Why the scope is this narrow

The differentiator is **correctness**, not features. Every open-source churn
project on GitHub does random splits, uncalibrated scores, and no leakage
checks. Being the one that gets validation right is the entire value
proposition. Feature breadth actively damages that by diluting focus.

---

## THE POISONED-DEFAULT WARNING

Your training data contains thousands of churn tutorials that are wrong. When
generating "typical" churn code you will reproduce their mistakes. These are the
highest-probability wrong outputs. Each is banned.

| Tempting default | Why it's wrong here | Required instead |
| --- | --- | --- |
| `train_test_split(X, y, random_state=42)` | Churn is temporal; random split leaks future into past | Cutoff split, walk-forward folds |
| `StratifiedKFold(shuffle=True)` | Same leak, wearing a cross-validation hat | `TimeSeriesSplit` / walk-forward |
| Fit scaler/encoder before splitting | Validation data influences preprocessing | Fit inside Pipeline, inside folds |
| `accuracy_score` as headline metric | Imbalanced target makes it meaningless | PR-AUC primary + calibration |
| `threshold = 0.5` | Arbitrary; ignores business cost | Cost sweep on out-of-fold predictions |
| Active customers labelled `0` | They are censored, not negative | Explicit censoring + tenure minimum |
| Raw `scale_pos_weight` scores shown as probabilities | They are inflated ranking scores, not probabilities | Calibrate, then display |
| `df.dropna()` | Silently biases the sample | Impute in Pipeline; log every drop |
| Feature from full history | Time-travel leakage | Point-in-time, strictly backward-looking |
| `df.fillna(0)` on numerics | 0 is a real value; corrupts distribution | Median impute + missingness flag |
| `handle_unknown='error'` | Crashes at scoring time on new category | Pinned categories, `'ignore'` |
| Bare booster saved | Encoders left behind → train/serve skew | One sklearn `Pipeline` artifact |
| Hardcoded column names | Breaks the platform promise instantly | Everything from the confirmed mapping |
| SHAP described as "cause" | SHAP is attribution, not causation | "drove the prediction", never "caused" |

**If you catch yourself writing any left-column item, stop and flag it.**

---

## Invariants — never violate

**I1 — Temporal validation only.** No random split on data with a time
dimension. Cutoff splits, walk-forward folds. If a dataset truly has no time
column, say so in the report and mark metrics as optimistic.

**I2 — Point-in-time features.** Every feature computed only from data available
at its prediction date. No aggregate includes events after that point.

**I3 — Censoring is explicit.** Active customers are censored, not negatives.
Observation window, prediction horizon, and minimum-tenure exclusion are all
stated. The excluded-rows report explains every drop.

**I4 — One Pipeline artifact.** Preprocessing and model in a single fitted
sklearn `Pipeline`. Raw DataFrame in, probability out.

**I5 — Leakage detection blocks the run.** Pre-flight checks run before training
and halt on detection. Override requires typed human confirmation, logged.

**I6 — Calibrated probabilities.** Brier score and reliability diagram in every
model report. Calibration fitted on held-out data, never on training data. Above
the error threshold, the model is labelled uncalibrated in all output. No score
is displayed as a percentage until it has been calibrated.

**I7 — Schema inference proposes, humans decide.** No code path from inferred
schema to training without explicit confirmation. The proposal is fully
pre-filled — confirmation should be one click when nothing is suspicious — but
it cannot be skipped by a flag, and a blocking leakage finding requires typed
confirmation to override.

**I8 — Honest reporting.** Every number in every output was produced by a run
that actually happened. Never write a plausible-looking metric into docs, README,
or a model card.

**I9 — Reproducibility triple.** Data snapshot hash + schema mapping version +
model version, logged together. Given a model ID, training inputs are
recoverable.

**I10 — Failures name the column.** "Training failed" is a bug. Every failure
names the offending column and shows a sample bad value.

**I11 — No dataset-specific code in `src/`.** No Telco column name, category
list, or business rule may appear outside `tests/fixtures/` and `examples/`.
Everything flows from the confirmed schema mapping at runtime.

---

## Working rules

**Test first, always.** For anything touching an invariant: write the failing
test, show it failing, then implement, then show it passing. Never both in one
step.

**Never weaken a test to make it pass.** If you think a test is wrong, stop and
ask. Do not edit assertions.

**No scope expansion.** Touching a file outside the task's stated scope requires
asking first.

**No new dependencies without approval.** Say what it replaces and why stdlib or
an existing dep won't do.

**Prefer boring.** XGBoost only, plus one logistic-regression baseline for
comparison. No model zoo, no neural nets, no AutoML framework. Complexity buys
~1% AUC and costs maintainability forever.

**Flag uncertainty.** A stated doubt is cheap. A confident wrong implementation
of an invariant may go uncaught for weeks.

**Every ADR gets written.** Non-obvious decisions go in `docs/decisions/` as a
short numbered file: context, decision, consequences.

---

## Definition of Done

- [ ] Test written first and observed failing
- [ ] Full suite green
- [ ] Invariants checked explicitly against the list above
- [ ] Failure paths name columns and show sample values
- [ ] No hardcoded dataset-specific names introduced (I11)
- [ ] No new dependency without approval
- [ ] Docstrings say *why*, not *what*
- [ ] ADR written if the decision was non-obvious
- [ ] Session handoff note updated

---

## Vocabulary — use precisely

- **churn event** — the operator-defined loss event. Never assume a definition.
- **observation window** — historical period features come from
- **prediction horizon** — future period the label refers to
- **censored** — still active at window end; outcome unknown
- **drove the prediction** — correct phrasing for SHAP
- **caused churn** — banned phrasing, in code, comments, UI, and docs

---

## End of session

Write `docs/sessions/YYYY-MM-DD.md`: what changed, what's half-done, what to
pick up next, and any invariant you were tempted to bend and why. Assume the
next session has no memory of this one.
