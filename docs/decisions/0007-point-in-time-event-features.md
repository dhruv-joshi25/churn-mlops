# 0007 — Point-in-time event features, and why the window is strict

## Context

A snapshot table says a customer makes 15 logins a month. An event log says they
fell from 15 to 3, and the fall is the signal. S6 is where accuracy actually
comes from — better features, not a better model.

The cost of getting it wrong is asymmetric and invisible. A feature computed
from data that postdates its prediction inflates every validation metric,
survives into the model artifact with no trace, and cannot be detected
downstream. The model works in testing and does nothing in production, and there
is no error message anywhere in that sequence.

## Decision

**The window is strict: `timestamp < prediction_date`, never `<=`.** An event
dated on the prediction date has not finished happening at the moment of
scoring. A same-day purchase is the most tempting boundary case precisely
because it is the most predictive, which is the argument for excluding it rather
than against.

**Every feature carries its provenance.** `FeatureMatrix.source_dates` records,
per row, the latest event timestamp that fed it, and
`assert_no_time_travel` is called on those timestamps **inside**
`build_features`, not only in the test suite. S6 asked for this specifically.
A test can be deleted or skipped; a call in the generation path fails the build
that produces the features.

**Feature groups are selectable on one implementation.** The benchmarking hook
S6 asks for (`groups=("counts",)`) filters this code path rather than providing
a second, simpler one. Two implementations would drift, and the comparison
between them would then be measuring the drift rather than the features.

**The engagement slope fits over the span actually observed, not the full
ninety days.** This was a real bug caught by the fixture: padding the empty time
before a customer's first event with zeros puts their decline in the middle of a
series that is flat on both sides, and a straight line through that is
horizontal — a leaving customer scores as steady. It also penalises every recent
signup for not having existed. The fit now runs from the customer's earliest
event inside the window to the prediction date. Verified by sign on three
shapes: declining −0.021, growing +0.017, steady −0.001.

**Excluded future events are counted and reported, not silently dropped.**
`n_events_excluded_as_future` and a warning say how many events fell on or after
their prediction date. That number is *expected* to be non-zero and is not a
data-quality problem; reporting it is how an operator can tell the windowing ran
at all.

## Consequences

**Cost.** Features are computed per customer per prediction date in Python
rather than vectorised across the whole log, because the correctness property —
each row sees only its own past — is easy to state that way and easy to get
wrong in a vectorised rewrite. If this becomes a bottleneck, the rewrite needs
`source_dates` as its correctness test, and that test already exists.

**The parser is the slow part, not this.** Reading 541,910 rows of real
transaction data through `read_table` takes about 60 seconds. It is a
non-streaming parser by design (ADR on `MAX_FILE_BYTES` covers the memory side),
and the time is spent on per-column type inference that a faster reader would
skip and get wrong. Worth revisiting when there is evidence it matters.

**No churn definition yet.** This module builds features from an event log; it
does not decide what churn *is*. Real transactional data has no churn column —
verified on the UCI Online Retail dataset, where churnkit's own inference
correctly reports `target=None` and says the churn definition must be supplied.
Deriving churn from inactivity ("no purchase in 90 days") is ROADMAP 2.2 and is
the next thing this needs.
