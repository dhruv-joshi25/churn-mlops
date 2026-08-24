# 2026-08-24 — S4: schema inference and leakage detection

## What changed

`src/churnkit/ingest/infer.py`, built test-first against
`tests/fixtures/leaky/MANIFEST.md`. 141 tests → 184. Coverage on `ingest/` is
93% overall, 95% on the new module.

**Part A — roles.** Every column gets one of eight roles with a confidence and a
sentence of reasoning: identifier, numeric, categorical (low or high
cardinality), datetime, free text, constant, target candidate. Target, ID and
timestamp columns are proposed and ranked, and exclusions carry reasons.

**Part B — leakage.** Five rules: single-column AUC, null-mask correlation,
post-window datetimes, name patterns, and constant-within-class. Two severities,
with a sample-size tier on the statistics. ADR 0005 has the full reasoning.

**I7 is a type boundary, not a convention.** `SchemaProposal.to_mapping()`
always raises. The only route to a `ConfirmedMapping` is `confirm()`, which
requires the target named explicitly, and where a blocking finding exists
requires the operator to type `I ACCEPT THE LEAKAGE RISK` in full. The override
is recorded on the mapping with the findings it overrode (I5).

## Verified beyond the suite

Run against the real Telco CSV — 7043 rows, 21 columns, a dataset the module has
never seen and shares no code with (I11). It proposed `Churn` as the target at
0.9 confidence, ranked the binary-but-unrelated `SeniorCitizen` at 0.5 below it,
found `customerID` as the identifier, classified all 18 remaining columns
correctly, and produced **zero leakage findings**. It also warned that the file
has no timestamp column, so temporal validation is impossible and any metric
from it is optimistic — which is true of Telco and is exactly what I1 asks for.

## Three bugs the fixtures caught that review would not have

1. **`last_` in the name-pattern list.** An early draft flagged `last_seen_on`.
   `last_login` and `last_payment` are the most valuable legitimate churn
   features there are; flagging them fires on nearly every real dataset and
   teaches operators to ignore warnings. Removed, and ADR 0005 records why.

2. **Globally-constant columns read as perfect leaks.** A column with one value
   throughout is trivially constant within every class. `tenant_label` in
   `07_roles.csv` exposed it. The rule now requires the column to vary overall.

3. **Two of my own fixtures did not test what they claimed.** The small-sample
   fixture had *perfect* separation rather than the 0.972 AUC its docstring
   described, and the demographic fixture built `marital_status` perfectly
   aligned with the target, making it a genuine leak instead of the benign name
   it was meant to be. Both were rewritten; the AUC is now computed and asserted
   rather than assumed.

## An invariant I was tempted to bend

`test_layout_guards.py` bans `.dropna(`, `.fillna(` and `errors="coerce"`
anywhere under `src/churnkit/ingest/`. My first implementation used all three —
`dropna()` to compute statistics over present values, and `to_numeric(...,
coerce)` to turn the target into 0/1. The uses were arguably legitimate, since
they compute over data rather than altering it, and widening the guard to allow
`infer.py` would have taken one line.

I did not. Everything was rewritten to explicit `notna()` masks, and the target's
positive class is now found by taking the larger of its two distinct values —
which works for `0`/`1` and for `"No"`/`"Yes"` alike, and cannot silently turn an
unexpected spelling into a missing value the way coercion would. The guard
stayed as written. It is worth noting that the rewrite produced better code than
the version that needed the exemption.

## Pick up next

**S5 — temporal validation**, the core differentiator. It has a hard dependency
this session surfaced: `infer_schema` proposes a timestamp column and says
plainly when there is not one, and S5 needs both paths. Telco has no timestamp,
so the reference dataset exercises the "no time column, metrics are optimistic"
branch rather than the walk-forward one — S5 will need a fixture with a real
time dimension to test the path that matters.

Nothing is half-done. `observation_end` is a parameter of `infer_schema` and
nothing supplies it yet; that becomes real in S5 when the window is defined.
