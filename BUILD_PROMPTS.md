# Build Prompts — Claude Code, Session by Session

Seventeen sessions from empty repo to releasable v0.1. One session per sitting.
Do not merge sessions; scope creep is how this fails.

---

## Before you start

1. `CLAUDE.md` must be in the repo root. It is the only thing that survives
   between sessions.
2. Git initialised, first commit made.
3. Read every diff before accepting. Claude writes; you are accountable for what
   ships.

**The single discipline that matters:** never accept "done" without having seen
the test fail first. If Claude shows you only passing tests, it may have written
the test to fit code it already wrote. Ask: *"show me that test failing against
the previous commit."*

---

## S0 — Every session starts here

```
Read CLAUDE.md in the repo root, then docs/sessions/ (most recent three files).

Tell me in five lines or fewer:
1. What the last session completed
2. What is half-finished
3. Which invariants this session's work will touch
4. Your plan, as numbered steps
5. Anything in CLAUDE.md you think is wrong, unclear, or that you disagree with

Write no code yet. Wait for my go-ahead.
```

That fifth question matters. If Claude never pushes back on anything across
seventeen sessions, it is agreeing too easily and you should be more suspicious of
the rest of its output.

---

## S1 — Project skeleton and CI

```
Read CLAUDE.md. Set up the project skeleton. No business logic this session.

BUILD:
- pyproject.toml, package name `churnkit`, Python 3.11+, src layout
- Pin: pandas, numpy, scikit-learn, xgboost, shap, pydantic, typer, jinja2
- Dev deps: pytest, pytest-cov, ruff, mypy
- ruff + mypy configured strictly; mypy strict on src/
- .github/workflows/ci.yml: lint, type-check, test, coverage gate at 80%
- Makefile: install, test, lint, typecheck, run
- .gitignore covering data files, models, .env, caches
- A single smoke test so CI has something to run

CONSTRAINT: no dependency beyond the list above without asking me first.

Then explain your pyproject choices in three lines and stop.
```

---

## S2 — Nasty-CSV fixture corpus

**Do this before writing the parser.** Building the adversary first means you
cannot unconsciously write a parser that only handles what you thought of.

```
Read CLAUDE.md. This session builds ONLY test fixtures. No parser code.

CREATE tests/fixtures/nasty/ with at least 15 deliberately awful CSVs, each
isolating one failure mode:

1. Semicolon-delimited
2. Tab-delimited with quoted fields containing tabs
3. UTF-8 with BOM
4. Latin-1 encoded, accented characters
5. Windows-1252 with smart quotes
6. Dates as DD.MM.YYYY
7. Dates as Excel serial integers
8. Ambiguous dates (03/04/2024 — could be either format)
9. Mixed null spellings: "", NA, N/A, null, NULL, None, -, --, ?
10. Currency-polluted numerics: "$1,234.56", "₹1.2L", "45%"
11. Duplicate column names
12. Real header on row 3, junk above it
13. Trailing empty rows and columns
14. Whitespace-padded headers and values
15. Ragged rows (inconsistent field counts)

For each, also write tests/fixtures/nasty/MANIFEST.md: filename, what's wrong,
what correct parsing should produce, and what the parser must NOT do silently.

These fixtures define correctness for the next session. Make them genuinely
hard — assume the parser author is trying to pass them cheaply.
```

---

## S3 — The parser

```
Read CLAUDE.md and tests/fixtures/nasty/MANIFEST.md.

BUILD src/churnkit/ingest/reader.py.

TEST FIRST: write tests against every fixture from S2. Show me the full suite
failing before you implement anything.

REQUIREMENTS:
- Sniff delimiter and encoding; never assume
- Detect the real header row
- Parse dates across the formats in the fixtures
- AMBIGUOUS DATES: do not guess. Return a flag requiring disambiguation.
- Recognise all null spellings from the manifest
- Strip currency symbols and separators from numerics
- Deduplicate column names deterministically

RETURN a ParseResult dataclass: DataFrame, per-column parse stats
(n_parsed, n_failed, sample_failures), detected encoding and delimiter, and a
warnings list.

BANNED in this module: dropna, fillna, and any silent coercion. A value that
cannot be parsed is counted and reported, never quietly replaced.

Invariant I10 applies to every failure path.
```

---

## S4 — Schema inference and leakage detection

```
Read CLAUDE.md, especially I5 and I7.

BUILD src/churnkit/ingest/infer.py.

PART A — role inference per column: identifier, numeric, low-cardinality
categorical, high-cardinality categorical, datetime, free text, constant,
target-candidate. Each with a confidence score.

Also propose: churn target column (ranked candidates with reasoning), customer
ID column, timestamp column (required for I1), and columns to exclude with
reasons.

PART B — leakage detection. Flag as probable leakage:
- Any single column reaching > 0.95 AUC against the target alone
- Columns whose null-mask correlates > 0.9 with the target
- Datetimes with values after the observation window ends
- Name patterns: cancel*, churn*, terminat*, end_date, exit*, status, reason*,
  closed*, refund*
- Columns constant within each target class

BEFORE IMPLEMENTING: explain your thresholds and your false-positive reasoning.
I want to review them.

OUTPUT is a SchemaProposal. It is a proposal only. Write a test that attempts to
trigger training directly from a proposal and asserts it raises.

TEST FIRST, including a fixture with a planted `cancellation_date` leak that
must be caught and must block.
```

---

## S5 — Temporal validation (the core differentiator)

```
Read CLAUDE.md, especially I1, I2, I3.

WARNING: your training data is saturated with churn code using train_test_split
with random_state. That code is wrong for this problem. If you find yourself
importing train_test_split in this module, stop and tell me.

BUILD src/churnkit/validation/temporal.py:

1. cutoff_split(df, date_col, cutoff) — train strictly before cutoff, evaluate
   on the labelled window after
2. walk_forward_folds(df, date_col, n_folds, horizon) — expanding window
3. label_with_censoring(df, obs_window, horizon, min_tenure) — returns labels
   PLUS an excluded-rows report explaining every drop (I3, I10)
4. assert_no_time_travel(features, prediction_dates) — raises if any feature's
   source data postdates its prediction date

THEN add a structural guard: a test that scans src/ and fails the suite if
train_test_split is imported anywhere outside an explicit allow-list. Written
instructions get forgotten across sessions; a failing build does not.

TEST FIRST. Include a synthetic dataset where random splitting yields materially
higher AUC than temporal, and assert the temporal figure is what gets reported.

Explain your design before writing it. This module is the whole project.
```

---

## S6 — Event-log features

```
Read CLAUDE.md, especially I2.

CONTEXT: snapshot tables lose the signal. A customer at 15 logins/month is
healthy; one who fell from 15 to 3 is leaving. This is where real accuracy comes
from — better features, not a better model.

BUILD src/churnkit/features/events.py.

INPUT: long-format event log (customer_id, timestamp, event_type, value) and a
set of prediction dates.

GENERATE per customer per prediction date, using ONLY events strictly before it:
- Rolling counts over 7/30/90-day windows, per event type
- Window-over-window deltas, absolute and ratio
- Recency: days since last event, per type
- Frequency: events per active day
- Monetary: sum and mean of value, per window
- Engagement slope: linear trend over trailing 90 days
- Longest inactivity gap in window
- Threshold flags: usage down >30% month-over-month

I2 IS THE ENTIRE POINT. Call assert_no_time_travel inside the generation path,
not only in tests.

TEST FIRST with a fixture where a naive implementation would swallow a future
event; assert yours excludes it.

Also expose a hook so I can benchmark snapshot-only vs snapshot+events later.
```

---

## S7 — Training, calibration, threshold

```
Read CLAUDE.md, especially I4, I6, I9.

BUILD src/churnkit/training/train.py.

1. Single sklearn Pipeline artifact (I4) — raw DataFrame in, probability out
2. Temporal validation only, via validation/temporal.py
3. scale_pos_weight computed per dataset, never hardcoded
4. Calibration: CalibratedClassifierCV, isotonic above 1000 rows else sigmoid,
   fitted on a held-out TEMPORAL slice — never on training data (I6)
5. Metrics: PR-AUC primary, ROC-AUC, Brier score, reliability-diagram data
6. Cost-based threshold swept on out-of-fold TRAINING predictions only; costs
   come from config, never hardcoded
7. Segment metrics: PR-AUC and calibration by tenure band, plan tier if present,
   revenue decile — with warnings on underperforming segments
8. Log the reproducibility triple (I9)

Under 500 labelled rows: refuse to emit a headline metric; return a warning that
the dataset is too small for reliable validation.

TEST FIRST. Assert: no random split reachable, calibration fitted only on
held-out data, threshold from out-of-fold predictions, small-data refusal fires.
```

---

## S8 — Scoring, SHAP, and actions

```
Read CLAUDE.md. Note the banned causal vocabulary.

BUILD src/churnkit/training/score.py and src/churnkit/reporting/actions.py.

SCORING: load Pipeline → score customers → output a CSV with customer_id,
calibrated probability, risk band, top-4 drivers with direction and magnitude,
suggested action.

SHAP: TreeExplainer on the model step. Map one-hot column names back to readable
labels ("Contract_Month-to-month" → "Month-to-month contract"). Nobody wants raw
column names in output.

ACTIONS: rule engine mapping drivers to suggested interventions, configured in
YAML, not hardcoded.

LANGUAGE REQUIREMENT: every string a user sees — CSV headers, report text, log
lines — must use "drove the prediction" framing. The words "caused", "because
of", and "reason for churn" are banned in user-facing output. Write a test that
greps the codebase for banned phrases in user-facing strings and fails on a hit.
```

---

## S9 — Model card and HTML report

```
Read CLAUDE.md, especially I8.

BUILD src/churnkit/reporting/model_card.py — generates a standalone HTML report
per trained model, using jinja2.

MUST CONTAIN:
- Intended use and explicitly out-of-scope uses
- Training window, prediction horizon, censoring rule applied
- Row counts: total, used, excluded (with reasons)
- PR-AUC, ROC-AUC, Brier, reliability diagram
- Per-segment performance table with underperformance warnings
- Chosen threshold and the cost assumptions behind it
- The leakage checks that ran and their results
- Known limitations
- Explicit statement: drivers are correlational attributions, not causal
  claims; recommendations are hypotheses requiring an A/B test

I8 IS ABSOLUTE. Every number comes from the actual run. No placeholder values,
no illustrative examples that could be mistaken for results. If a metric could
not be computed, the report says so rather than omitting the row.

TEST: assert the report renders from a real training run and that every numeric
field traces to the metrics object.
```

---

## S10 — CLI and packaging

```
Read CLAUDE.md.

BUILD the operator-facing surface. Typer CLI:

  churnkit inspect data.csv          → parse stats + schema proposal
  churnkit confirm proposal.yaml     → human confirmation step (I7)
  churnkit train --config config.yaml
  churnkit score --model <id> --input new.csv --output scored.csv
  churnkit report --model <id>       → HTML model card

Plus: a single config.yaml holding churn definition, observation window,
prediction horizon, minimum tenure, cost-of-offer, value-of-save, action rules.

Plus: Dockerfile and docker-compose.yml so `docker compose up` gives a working
environment against a mounted data directory.

Plus: examples/ with a small synthetic dataset and a walkthrough that runs
end to end in under two minutes.

The confirm step is not optional and must not be bypassable by a flag.
```

---

## S11 — Benchmark harness

**This is what makes the positioning credible. Do not skip it.**

```
Read CLAUDE.md, especially I8.

BUILD benchmarks/ — a harness running the full pipeline across multiple public
churn datasets (Telco, bank churn, and 2-3 others you identify).

FOR EACH DATASET REPORT:
- Random-split AUC vs temporal-split AUC (the gap is the headline finding)
- Snapshot-only features vs snapshot+event features where events exist
- Calibration before and after
- Runtime and peak memory
- Any dataset where the pipeline failed or degraded, and why

OUTPUT a markdown results table, generated from actual runs, timestamped, with
the commit hash.

I8 IS THE WHOLE POINT OF THIS SESSION. If a dataset is unavailable or a run
fails, report the failure. Do not fill the gap with a plausible number. I would
rather have four honest rows than six with one invented.

Include the raw result JSON in the repo so the table is verifiable.
```

---

## S12 — MLflow tracking and model registry

```
Read CLAUDE.md, especially I9.

BUILD src/churnkit/training/tracking.py.

Every training run logs to MLflow: params, all metrics from S7, the Pipeline
artifact, the reliability-diagram data, the segment table, the schema mapping
version, and the data snapshot hash (I9 — the reproducibility triple).

Registry: model versions with stages (Staging, Production, Archived). Promotion
requires an explicit CLI command with typed confirmation — no code path may
promote automatically.

Backend: sqlite + local artifact store by default, so `docker compose up` works
with no external services. Postgres optional via config for operators who want
it.

Add: `churnkit models list`, `churnkit models compare <a> <b>`,
`churnkit models promote <id>`, `churnkit models rollback`.

The compare command must surface the metric that matters — PR-AUC and
calibration error side by side, not accuracy.

TEST FIRST. Assert the reproducibility triple is recoverable from a model ID
alone, and that no automatic promotion path exists.
```

---

## S13 — Serving API

```
Read CLAUDE.md, especially I4.

BUILD src/churnkit/serving/ — FastAPI service loading the Production model from
the registry.

ENDPOINTS:
  GET  /health            → liveness + loaded model version; returns "degraded",
                            never 500, when no model is registered
  POST /predict           → one customer; probability, band, drivers, action
  POST /predict/batch     → up to 1000; drivers off by default for speed
  POST /reload            → pull current Production model without restart
  GET  /model/card        → the S9 model card as JSON

REQUEST VALIDATION: generated from the CONFIRMED schema mapping, not hardcoded.
Unlike a fixed-dataset project, the accepted fields differ per deployment. Build
the pydantic model dynamically from the stored mapping.

The loaded object is a single Pipeline (I4). Raw fields in, probability out.

Model cache with LRU eviction — an operator may have several model versions.

TEST FIRST including: unknown category handled without crashing, missing
optional field imputed, /reload picks up a newly promoted version.
```

---

## S14 — Web UI (upload → confirm → results)

```
Read CLAUDE.md, especially I7.

BUILD a Streamlit app covering the operator's full loop without the CLI.

FLOW:
1. Upload CSV (and optional event log)
2. Show parse report: what was detected, what failed, warnings
3. Show schema proposal with confidence scores — EDITABLE
4. Show leakage findings prominently; blocking findings cannot be dismissed
   without typed confirmation (I5, I7)
5. Confirm mapping → trigger training
6. Live training status
7. Results: scored table, sortable by risk, with drivers and suggested action
8. Model card rendered inline
9. Download scored CSV

The UI calls the API over HTTP. It must NOT import the model directly — that
would create a second inference path and defeat the architecture. Write a test
asserting the UI package does not import churnkit.training or churnkit.serving
internals.

LANGUAGE: every user-facing string uses "drove the prediction" framing. The
causal-language grep test from S8 must cover this package too.
```

---

## S15 — Drift monitoring

```
Read CLAUDE.md, especially I8.

BUILD src/churnkit/monitoring/.

TRACK, per scoring run:
- Input feature distributions vs the training distribution (PSI or KS per
  feature, with a configurable alert threshold)
- Prediction distribution shift over time
- Missingness rate changes per column
- New categorical levels not seen in training
- When ground-truth outcomes arrive: realised PR-AUC and calibration error vs
  the values reported at training time

ALERTS: written to a status file and surfaced in the UI and CLI. No paging
integration in v0.1 — keep it boring.

BE HONEST ABOUT WHAT THIS CAN AND CANNOT DO. Feature drift is detectable
immediately. Performance drift is only detectable once outcomes are observed,
which lags by the prediction horizon. The monitoring report must state this
distinction plainly rather than implying live accuracy tracking (I8).

TEST FIRST with a fixture where a feature distribution is deliberately shifted;
assert detection fires. Include a second fixture with normal variation; assert
it does NOT fire.
```

---

## S16 — Retraining loop

```
Read CLAUDE.md, especially I1, I3, I7.

BUILD `churnkit retrain` — retrains on newer data as it accumulates.

REQUIREMENTS:
- New cutoff date moves forward; observation window and horizon shift with it
  (I1 — this is still temporal, not a random refit)
- Censoring rule reapplied at the new window (I3)
- Schema mapping reused from the confirmed version; if incoming columns no
  longer match, FAIL LOUDLY naming the mismatched columns (I10) rather than
  silently coercing
- New model lands in Staging, never Production (I7)
- Automatic comparison against the current Production model on a common
  temporal holdout, with a recommendation — but the promote decision stays human
- Scheduling: a documented cron/systemd example, not a built-in scheduler.
  Operators already have schedulers; do not reinvent one.

TEST FIRST: assert retraining on shifted data produces a Staging model, never
Production; assert a schema mismatch fails loudly with column names.

Write an ADR on why promotion stays manual even in the retraining loop.
```

---

## S17 — Release readiness

```
Read CLAUDE.md.

PREPARE for public release:

- LICENSE: AGPL-3.0 (chosen so nobody can host this commercially without
  contributing back — write the ADR explaining this)
- SECURITY.md: disclosure contact, response commitment
- CONTRIBUTING.md, issue templates, PR template
- README rewritten: outcome-first opening, who it's for, honest limitations
  section, benchmark table from S11, quickstart under 10 lines
- docs/decisions/ — ensure an ADR exists for: XGBoost-only, temporal-only
  validation, human confirmation on schema, self-hosted single-tenant, AGPL
- Run gitleaks over full history; report anything found
- Verify no customer data, real or synthetic-but-realistic, is committed

README LIMITATIONS SECTION IS MANDATORY and must state at minimum: needs a
timestamp column, needs 500+ labelled examples, binary churn only, no real-time
scoring, no causal uplift in v0.1, tabular only.

Overclaiming here undoes everything the previous eleven sessions built.
```

---

## Adversarial passes — run these, they are your only review

Solo means no second pair of eyes. These substitute. Run R1 and R2 at the end of
every session; R3 after S4 and again after S7; R4 and R5 before release. Run R1 again
after S16 — the retraining loop is where invariants quietly rot.

### R1 — Invariant audit
```
Review this codebase adversarially. Assume the author was competent but rushed
and that at least one invariant in CLAUDE.md is violated.

Go through I1-I10 one at a time. For each, cite the file and line that enforces
it, or state that nothing does. Do not accept "handled in the pipeline" without
pointing at code.

Then: which invariant is most weakly enforced, and what is the smallest change
that would make it robust? Report only — fix nothing yet.
```

### R2 — Poisoned-default hunt
```
Search the codebase for every anti-pattern in the CLAUDE.md poisoned-default
table. Show file, line, and why it's wrong in this context.

Check especially: train_test_split anywhere in src/, fit() on preprocessing
outside a Pipeline, accuracy_score, literal 0.5 thresholds, dropna, fillna(0),
aggregates over full history, and causal language in comments, docstrings, CLI
output, or docs.

Report all findings before changing anything.
```

### R3 — Leakage red-team
```
You are trying to make this system produce a falsely excellent model.

Construct three datasets that pass current leakage detection but yield inflated
metrics. Think about columns encoding the outcome indirectly, subtly post-hoc
timestamps, and aggregates computed after the fact.

For each, report whether the system caught it. For misses, propose a detection
rule that would catch it without excessive false positives.
```

### R4 — Fresh-eyes onboarding
```
You have never seen this repo. You are an engineer at a company evaluating it.
Follow the README exactly. Report every point where you get stuck, confused, or
must guess.

Use no knowledge from previous sessions. If a step fails, report the failure
rather than working around it.
```

### R5 — Documentation truth check
```
Every claim in README.md, docs/, and any model card must trace to code or to a
benchmark run that actually happened.

Go claim by claim. Mark each VERIFIED (cite the code or result file),
UNVERIFIED (plausible but not demonstrated), or FALSE.

Any number not produced by a real run gets flagged and removed. Do not fill gaps
with plausible values.
```

---

## What Claude will reliably get wrong

Watch for these yourself. They slip past even careful sessions.

1. **Reverting to random splits** in any new module that feels like "normal ML
   code", even after being told not to. Re-check after every ML commit. This is
   why S5 builds a structural guard rather than relying on instructions.

2. **Calibration fitted on training data** — produces beautiful, meaningless
   reliability diagrams.

3. **Test weakening.** When a test won't pass, an agent sometimes adjusts the
   assertion. Diff your test files, not just source.

4. **Fabricated benchmark numbers.** The most damaging failure, because it is
   invisible until an outsider checks and by then you have published it.

5. **Causal language in user-facing strings** while the code stays careful.

6. **Silent row drops** — a `.dropna()` added to fix a crash, quietly biasing
   every model afterwards.

7. **Model-zoo creep.** Agents love adding LightGBM and CatBoost. ~1% AUC,
   double the maintenance, forever.

8. **Losing the reproducibility triple** during refactors.

9. **Skipping the session handoff note**, which makes the next session start
   blind and redo work.

10. **Agreeing with everything.** If Claude never pushes back across a whole
    session, treat that as a signal to check its work harder, not as a sign
    things are going well.
