# 0004 — Guarding /reload, and replacing the wildcard CORS origin

## Context

A security pass over the branch before pushing to `main` turned up two findings
in `churnkit/reference/api/main.py` that share one root cause.

`POST /reload` swaps the model the API serves. It was unauthenticated, took no
body, and sat behind `allow_origins=["*"]` with `allow_methods=["*"]` and
`allow_headers=["*"]`. That combination is enough for any page the operator
happens to have open in another tab to reload the served model on their behalf:
a `fetch` with `mode: "no-cors"` and no custom headers is a *simple request*, so
the browser sends it without a preflight and the server acts on it. The attacker
cannot read the response, which is exactly why this matters — the effect is
server-side and invisible to the operator.

The wildcard origin itself was the second finding. It was flagged in a code
comment as "tighten before any public deployment", which is the kind of note
that survives to production. It is worth being precise about its severity:
`allow_credentials` was never set, so it defaulted to `False`, and the dangerous
`"*"` + credentials pairing never existed here. On its own the wildcard leaks
read access to responses, and there is nothing behind this API that a reader
could not also request directly.

The complication is that CLAUDE.md and ADR 0001 say this product has no
authentication and is not to grow any. That is a deliberate scope boundary, and
a `/reload` token would sit uncomfortably close to it.

## Decision

**Treat this as CSRF, not as authentication, and fix it with the two-part
defence that CSRF actually calls for.**

1. **An origin allowlist replaces `"*"`.** `CORS_ORIGINS` in `churnkit/config.py`
   reads a comma-separated environment variable, defaulting to
   `http://localhost:8501` and `http://127.0.0.1:8501`. `allow_methods` and
   `allow_headers` are narrowed to what the API actually uses.

2. **`/reload` requires a custom header, `X-Churnkit-Admin`.** The value is
   ignored and is explicitly not a secret. What does the work is that it is a
   *custom* header: a browser will not attach one to a cross-origin request
   without first passing a preflight, and an unlisted origin fails that
   preflight. This is the OWASP custom-request-header defence.

Neither half works alone, which is why both landed together. An allowlist
without the header requirement still admits simple requests, since CORS governs
who may *read* a response, not who may *send* one. The header requirement
without an allowlist is defeated by the wildcard preflight approving it.

The shipped Streamlit portal is unaffected in substance: it calls the API
server-side with `requests`, so no browser request ever crosses an origin. Its
reload button now sends the header. The portal writes the header name out as a
literal rather than importing `churnkit.config`, because the import guard in
`tests/test_layout_guards.py` fences the UI to presentation modules; a new test
in that file pins the literal to `config.ADMIN_HEADER` so the copy cannot drift.

This adds no user accounts, no sessions, no `tenant_id`, and no secret to
manage, so it stays inside ADR 0001.

## Consequences

**What this defends against.** A page the operator visits while the deployment
is running can no longer reload the model underneath them, and can no longer
read this API's responses from an unlisted origin.

**What it explicitly does not defend against, and must not be mistaken for.**
Anyone who can reach the port can still send the header with `curl` and reload
the model. There is no authentication here and this ADR does not add any. The
protection is against *browsers being used as a confused deputy*, which is the
realistic threat when the deployment listens on the operator's own machine. A
deployment exposed to a hostile network needs a reverse proxy in front of it —
that has always been true of this API and remains true.

**Operational.** An operator serving the portal from a non-default origin must
set `CORS_ORIGINS`, or their own browser-side page gets blocked. The variable
was already documented in `.env.example` and is now actually read. Anyone
driving `/reload` from a script must add the header; the 403 names it.
