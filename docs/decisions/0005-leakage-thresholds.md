# 0005 — Leakage thresholds, and what they cost on clean data

## Context

`BUILD_PROMPTS.md` S4 specifies five leakage rules and asks for the thresholds
to be justified before implementation. The numbers it gives — 0.95 single-column
AUC, 0.9 null-mask correlation, a nine-entry name-pattern list — are a starting
point, not a specification, because a threshold without a false-positive budget
is just a number.

The asymmetry that drives everything here: **over-flagging is recoverable and
under-flagging is not.** I5 gives the operator a typed override, so a false
positive costs them a sentence of reading. A missed leak costs a model that
scores 0.99 in validation, ships, and predicts nothing — and nothing downstream
will catch it, because every metric will look excellent.

That argues for flagging aggressively. What stops it collapsing into "flag
everything" is that an operator who sees warnings on clean data learns the
warnings are noise, and overrides the real one when it arrives. A detector with
no false-positive budget has merely moved the failure from the model to the
human.

## Decision

**Two severities.** `blocking` halts the run and needs a typed override.
`warning` is reported prominently and does not halt. The rule is that blocking
means "we are confident", not "we noticed something".

**A sample-size tier on every statistic.** A borderline statistic on a small
sample is luck. With 40 rows and a 25% base rate there are 10 positives, and
single-column AUC has enormous variance at that size — a legitimately good
feature reaches 0.95 by chance. So:

| Evidence | Blocking |
| --- | --- |
| AUC ≥ 0.99, or exact separation | always |
| 0.95 ≤ AUC < 0.99 | only with ≥ 50 rows in the smaller class |
| null-mask \|r\| ≥ 0.99 | always |
| 0.90 ≤ \|r\| < 0.99 | only with ≥ 50 rows in the smaller class |
| constant within each class | always |

Perfect separation is exempt from the sample-size tier because no amount of
small-*n* luck produces a flawless ordering of a genuine feature.

**The name-pattern list is split.** `cancel`, `churn`, `terminat`, `exit`,
`refund`, `closed`, `lost`, `attrit` and `deactivat` each describe an event that
happens *because* the customer left, so they block on the name alone. `status`,
`reason`, `end_date` and `final_` warn only, and escalate to blocking when a
statistical rule fires on the same column — because `status` matches
`marital_status` and `employment_status`, `reason` matches `reason_for_signup`,
and an `end_date` may be a contract end known at signup.

**`last_` is deliberately absent from both lists.** S4's spec does not include
it and an earlier draft of this module added it. That was wrong: `last_login`,
`last_payment` and `last_seen` are the most valuable legitimate churn features
there are, and flagging them fires on nearly every real dataset. The
`04_post_window_dates.csv` fixture caught it, which is what that corpus is for.

**Constant-within-class requires the column to vary overall.** A column with one
value throughout is trivially constant inside every class. It is a useless
column, not a leak. `07_roles.csv` caught this one.

**AUC is computed on numeric columns only, never on datetimes.** A date that
separates the classes perfectly is caught by the post-window rule when a window
is known, and reported as unchecked when one is not. Ranking dates as if they
were magnitudes would flag every dataset where signup date drifts with cohort
quality, which is most of them.

**An unrun check is reported as skipped, never as a pass.** With no
`observation_end` the post-window rule cannot run at all. Reporting that as "no
leakage found" is the same silent-success failure the nasty-CSV corpus was built
around. The same applies to a date column the reader flagged as ambiguous: a
day-first/month-first misread pushes a date past the window spuriously, so the
check refuses to run and says why rather than guessing.

## Consequences

**The false-positive budget is measured, not asserted.**
`tests/fixtures/leaky/06_clean_baseline.csv` is 60 rows of genuinely predictive,
genuinely legitimate data, and a test asserts it produces zero blocking
findings. Verified on real data too: the Telco dataset, 7043 rows and 21
columns, produces zero findings, proposes `Churn` as the target over the
binary-but-unrelated `SeniorCitizen`, and correctly reports that it has no
timestamp column so its metrics will be optimistic (I1).

**Small datasets get warnings where large ones get halts.** An operator with
200 customers sees more warnings and fewer blocks than one with 200,000. That is
the honest reflection of how much the evidence supports — but it means a small
deployment leans harder on the human reading the proposal, which is what I7
already assumes.

**These thresholds are not tuned.** They are reasoned from what single features
can plausibly achieve on real churn data and checked against one clean fixture
and one real dataset. That is enough to ship a detector that does not cry wolf;
it is not enough to claim a calibrated false-positive rate, and no output should
imply one.
