# 0002 — MLOps is in scope for v0.1; multi-tenancy is not

## Context

An earlier scoping pass cut experiment tracking, serving, monitoring, and
retraining alongside multi-tenancy. That conflated two separate things.

## Decision

v0.1 includes the full MLOps loop: MLflow tracking and registry, a serving API,
drift monitoring, and operator-triggered retraining. It excludes multi-tenancy,
hosted deployment, and user accounts.

## Rationale

None of the MLOps components require multi-tenancy. A single operator running
one deployment still needs to compare runs, roll back a bad model, detect drift,
and retrain as data accumulates — arguably more so, since they have no data
science team to notice problems manually.

Multi-tenancy, by contrast, brings tenant isolation, auth, job orchestration,
and data-processor obligations under GDPR and India's DPDP Act 2023. Those are
genuinely separable and genuinely expensive.

## Consequences

- v0.1 grows from 12 to 17 build sessions
- The self-hosted promise is unaffected: data still never leaves the operator
- Multi-tenant hosting remains a v0.2 option on a proven core
- Promotion stays manual everywhere, including in the retraining loop
