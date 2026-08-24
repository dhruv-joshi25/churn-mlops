# 0001 — Self-hosted, single-tenant for v0.1

## Context

The original scope was a multi-tenant hosted platform where companies upload
customer data. That carries the full weight of tenant isolation, auth, job
queueing, and data-protection duties as a processor.

## Decision

Ship self-hosted and single-tenant. The operator runs it on their own
infrastructure against their own data.

## Consequences

- Customer data never reaches the maintainer; the maintainer is not a data
  processor under GDPR or India's DPDP Act 2023
- Removes tenant isolation, auth, and job orchestration from v0.1 entirely
- Better fit for the target user, who will not upload a customer list to an
  unknown vendor but will run a container they control
- Multi-tenant hosting remains possible later, on top of a proven core
- Distribution is harder: adoption requires the operator to deploy it themselves
