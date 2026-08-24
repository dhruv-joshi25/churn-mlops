# 0006 — Temporal validation, censoring, and a guard against the default

## Context

`BUILD_PROMPTS.md` S5 calls this module "the whole project", and the reason is
measurable rather than rhetorical. On a synthetic dataset where the churn driver
changes over time — which is what real churn data does — a random split reports
**AUC 0.759** for a model whose honest score is **0.521**. It is not slightly
optimistic. It is reporting a number for a task nobody will ever perform:
predicting the past from the future.

Every open-source churn repository does `train_test_split(X, y,
random_state=42)`. CLAUDE.md names it the first poisoned default, and the risk
is not that someone disagrees with the rule — it is that the rule is written
down in a file that a future session may not weigh heavily enough against a
training corpus saturated with the opposite pattern.

## Decision

**The cutoff is strict on the training side.** A row dated exactly on the cutoff
goes to evaluation. Boundary rows are the ones whose labels are most likely to
have been influenced by what happened just after the boundary, so the ambiguity
is resolved against the model rather than in its favour.

**Rows with no date are excluded from both sides, counted, and reported.** A row
that cannot be placed in time cannot be placed on a side of a temporal split,
and assigning one would put unknown data into training.

**Folds expand, they never shuffle.** `walk_forward_folds` grows the training
window and slides evaluation forward, which is how the model is actually used:
fitted on everything known so far, asked about what comes next. A
`KFold(shuffle=True)` is the same leak wearing cross-validation's clothes.

**Censoring is an exclusion, not a negative class.** This is the decision most
likely to be quietly undone later, so it is stated plainly: a customer with no
churn event who was *not observed for the full prediction horizon* has an
unknown outcome. `label_with_censoring` excludes them and counts them. Labelling
them 0 teaches the model that "we stopped collecting data" means "the customer
stayed", which is a model of the data pipeline rather than of churn (I3).

Four exclusion reasons, each counted with sampled identifiers (I10): started
after the window closed; below minimum tenure; churned inside the observation
window rather than the horizon; and censored. The counts reconcile against the
input row count, and a test asserts they do.

**`assert_no_time_travel` is an assertion, not a warning.** A feature computed
from data that postdates its prediction produces excellent validation metrics,
leaves no trace in the model artifact, and cannot be detected downstream. The
failure names the column, the number of rows, and the worst offending row with
both of its dates.

**A structural guard bans the default at build time.** `train_test_split`,
`ShuffleSplit`, `StratifiedShuffleSplit` and `shuffle=True` fail the suite
anywhere under `src/churnkit/`, with a named allow-list for genuine exceptions.
S5 asked for this specifically, and the reasoning is that written instructions
are forgotten between sessions while a failing build is not.

The guard strips comments *and string literals* before matching, via `ast`.
Without that, this very module — whose docstring explains why it never calls
`train_test_split` — would fail the guard against `train_test_split`. Prose
about a banned construct is fine; calling it is not.

## Consequences

**The guard immediately found a real violation.**
`churnkit/reference/train.py` calls `train_test_split(X, y, test_size=0.2,
stratify=y, random_state=42)` — the poisoned default verbatim, including the 42.
It is allow-listed rather than fixed, because the Telco reference dataset
genuinely has no time column (churnkit's own schema inference reports
`timestamp=None` for it) and I1 permits a random split in that case.

**But I1's other half is not satisfied, and this is an open finding.** I1 says
that where a dataset has no time column, the report must say so and mark the
metrics as optimistic. `train.py` does not: there is no mention of the missing
time dimension in its logged output, its model card, or the README figures
derived from it. The PR-AUC of 0.658 currently quoted in `ROADMAP.md` is
therefore an unmarked random-split number. Fixing it means touching
`reference/`, which is outside S5's stated scope, so it is recorded here rather
than done silently.

**Metrics will get worse when this is adopted.** Any figure this project has
published from a random split will fall when recomputed temporally, and the fall
is the point. Nothing in the codebase should present the two as comparable, and
a model card quoting a random-split AUC next to a temporal one would be the most
misleading thing this project could publish.
