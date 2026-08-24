# 0003 — The churnkit package layout, and quarantining the Telco code

## Context

`BUILD_PROMPTS.md` S1 specifies a `churnkit` package on a src layout with a
`pyproject.toml`, and every session from S3 onward writes to paths like
`src/churnkit/ingest/reader.py`. The repository had none of that: flat modules
under `src/`, imported as `src.train` and `src.api.main`, dependencies split
across `requirements.txt` and `requirements-ui.txt`, and CI that ran `pytest -q`
and nothing else.

Left unresolved this forks the project. Either the platform modules land at the
paths the build plan names while the existing code keeps its own import root, or
the plan gets rewritten around a layout it was not designed for. The first
option leaves two import styles in one tree; the second is a plan rewrite to
avoid an hour of mechanical work.

There is a second problem the layout has to answer. Invariant I11 says no
dataset-specific code in `src/`, and the existing code is Telco-specific
throughout: `schema.py` pins Telco's nineteen columns and their categories,
`data.py` compares the target against the literal `"Yes"`, `api/models.py`
declares every request field as a `Literal[...]` of Telco values. Moving it into
`src/churnkit/` unchanged would move an I11 violation into the platform package
rather than fixing anything.

## Decision

**One package, `churnkit`, on a src layout**, with everything moved into it in a
single change — modules, tests, the Streamlit app, both Dockerfiles, the
Makefile, and CI. No module keeps the old import root.

**`pyproject.toml` is the only dependency declaration.** Both requirements files
are deleted. Dependencies are split into extras rather than one list:

- core — pandas, numpy, pydantic, python-dotenv: what reading and describing an
  operator's data needs
- `ml` — scikit-learn, xgboost-cpu, shap, mlflow
- `serving` — `ml` plus FastAPI and uvicorn
- `ui` — Streamlit and requests, and deliberately **not** `ml`
- `cli` — typer and jinja2, reserved for S9 and S10
- `dev` — `serving` plus pytest, pytest-cov, ruff, mypy

The `ui` extra excluding the model stack is load-bearing, not tidiness. The
portal must reach a prediction only over HTTP; a second in-process inference
path would drift from the served one. Previously that was enforced by hand-
picking two files into the UI image. Now the UI environment simply has no
library capable of loading a model.

**The Telco implementation moves to `src/churnkit/reference/` and is fenced in
by a test.** `tests/test_layout_guards.py` fails if a Telco identifier appears
anywhere under `src/churnkit/` outside that directory. Platform modules are
clean from their first line, and each reference module is deleted as the
platform module replacing it lands.

**mypy strict applies to the platform; `churnkit.reference.*` is exempt.**
**The coverage gate is 80% on platform packages** (`churnkit/ingest` today),
with full-package coverage reported but not gated.

## Rationale

The quarantine is the honest option among three. Deleting the Telco code now
would remove the only working end-to-end path — training, registry, API, SHAP,
UI — before anything replaces it. Leaving it spread through the package would
mean I11 is violated everywhere and enforced nowhere, and the violation would
quietly grow: the next session that needs a column name has precedent for adding
one. Fencing it converts an unbounded violation into a bounded one that a
failing build defends, and makes the remaining work visible as a directory that
should be empty by v0.1.

Scoping mypy and coverage follows the same reasoning. Typing and covering code
with a known deletion date buys nothing; a green build that means something for
new code buys a lot. A gate set at a level the codebase already passes would
have been the alternative, and it would ratchet down as the honest thing to do —
better to gate the code the discipline is actually for.

## Consequences

- Imports are `from churnkit... import`; `python -m churnkit.reference.train`
  and `uvicorn churnkit.reference.api.main:app` replace the old entrypoints
- `pytest` works from a fresh clone without installing anything, via
  `pythonpath = ["src"]`; CI additionally installs the package so both paths are
  exercised
- Adding a Telco column name to a platform module now fails the suite
- The `reference/` directory is a progress bar: S4 removes `schema.py` and
  `data.py`, S13 removes `api/`, and when it is empty I11 holds outright
- Deviations from S1 as written, all deliberate: dependencies are split into
  extras rather than pinned as one list; mypy strict plus the coverage gate
  cover the platform packages rather than all of `src/`; and the Python floor is
  3.12 rather than S1's 3.11
- The Python floor is not a preference. `xgboost-cpu==3.4.1`, the version the
  registered reference model was trained with, publishes no wheel for 3.11, so
  the containers and CI — both pinned to 3.11 — could not install it. That was
  already true of the deleted `requirements.txt`, which means the API image and
  the CI Docker job had been failing before this change; moving the dependency
  declaration into pyproject is what surfaced it. Everything now builds on 3.12,
  which is also what the development virtualenv runs
