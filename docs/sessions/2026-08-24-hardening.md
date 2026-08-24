# 2026-08-24 — Security pass before pushing to main

## Why this session happened

The ask was to push `develop` and `main` together, re-verifying security first.
The push turned out to be a no-op — `main`, `develop`, `origin/main` and
`origin/develop` were all already at `81956cf` — so the session became the
security pass and the two fixes that came out of it.

## What the audit found

Clean on the things that matter most. No secret ever entered git history: the
only sensitive-shaped paths across all commits are `.env.example` (placeholders
only) and the synthetic `tests/fixtures/nasty/*.csv` corpus. No `eval`, `exec`,
`pickle`, `yaml.load`, `subprocess` or `shell=True` anywhere in `src/`, `app/`
or `tests/`. Both Dockerfiles drop to a non-root `appuser`. Every dependency is
`==` pinned. CI triggers on `pull_request`, not `pull_request_target`, so a
forked PR gets a read-only token and no secrets.

Four low-severity findings. Two were fixed this session; two were left, with
reasons, below.

## What changed

**`/reload` was drive-by CSRF-able.** It changes which model is served, took no
body, required nothing, and sat behind `allow_origins=["*"]`. A `no-cors` POST
from any page the operator had open would reload the model — a simple request,
so no preflight stood in the way. Fixed as CSRF rather than as authentication,
because ADR 0001 rules authentication out of scope: an origin allowlist
(`CORS_ORIGINS`, defaulting to the two localhost portal origins) plus a required
custom `X-Churnkit-Admin` header. Neither half works alone. ADR 0004 has the
full reasoning and, more importantly, states plainly what this does *not*
protect against — anyone who can reach the port can still `curl` it.

**`read_table` had no size cap.** It decodes a file whole and holds the raw
bytes, the decoded text, the split lines and the parsed records at once, so peak
memory ran to several times the file size with nothing bounding the input. Added
`MAX_FILE_BYTES` (256 MiB, chosen to comfortably admit the 200 MB Streamlit's
uploader accepts) and a `max_bytes` parameter, checked against `stat()` before a
byte is read. Verified live: a 286 MiB file is refused in under a millisecond at
102 MiB peak RSS, never loaded. The failure names the file, both sizes in bytes
and MiB, and how to raise the limit.

Both were written test-first and observed failing. 115 tests → 141.

## One thing worth knowing for next time

The portal cannot import `churnkit.config`, because `test_layout_guards.py`
fences the UI to presentation modules only. My first attempt imported
`ADMIN_HEADER` from config and the guard failed it — correctly. Rather than
touch that assertion, the header name is now written out literally in
`app/streamlit_app.py`, and a *new* test in the same file pins that literal to
`config.ADMIN_HEADER` so the duplicate cannot drift into a reload button that
silently 403s. That is the pattern to reuse whenever the UI needs to agree with
the API about something: duplicate the value, add a guard, leave the fence
alone.

## Left undone, deliberately

**CSV formula injection on the round-trip.** `app/streamlit_app.py` copies the
operator's raw cells into the downloadable scored CSV, so a cell starting `=`,
`+`, `-` or `@` round-trips and would execute as a formula in Excel. In a
single-tenant deployment this is the operator's own data returning to them, so
it was judged not worth changing the output. It becomes real the moment scored
files get forwarded to a third party — revisit it then, and note that the fix
(prefixing a quote) alters data the operator gave us, which needs a decision
rather than a quiet patch.

**`mlflow==2.16.2` is well behind current** and MLflow has a history of
tracking-server path-traversal CVEs. Here it is a local file store, so exposure
is nil. The standing rule is simply never to bind that server to a public
interface. A version bump is a separate change with its own test run.

## Pick up next

Nothing is half-done. The parser's next step is unchanged from
`2026-08-24-parser.md`; this session did not touch schema inference.

When the upload UI lands, `MAX_FILE_BYTES` should become the number the uploader
advertises and enforces, so the operator hits a clear limit in the browser
instead of a refusal after the transfer. The parameter is already there for it.
