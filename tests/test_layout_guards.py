"""Structural guards. Written instructions get forgotten; a failing build does not.

Two invariants are enforced here because both fail silently in review:

I11 — no dataset-specific code in the platform. The Telco reference
implementation is fenced into ``churnkit/reference``; this test fails if any of
its column names or category values leaks into a platform module.

The S3 ban on silent data loss — ``dropna``, ``fillna`` and coercion that
discards what it could not parse have no place in the ingest layer, where the
entire point is that every unparseable value is counted and reported.
"""

import re
from pathlib import Path

import pytest

from churnkit.reference import schema as telco

SRC = Path(__file__).resolve().parents[1] / "src" / "churnkit"
REFERENCE = SRC / "reference"

PLATFORM_MODULES = sorted(
    p for p in SRC.rglob("*.py") if REFERENCE not in p.parents and p.parent != REFERENCE
)

# Words that happen to be Telco column names but are ordinary churn vocabulary.
# The platform is expected to say "tenure" (CLAUDE.md itself asks for tenure-band
# segment metrics) without that meaning it has learned Telco's schema.
GENERIC = {"tenure", "gender", "Partner", "Dependents", "Contract"}

TELCO_IDENTIFIERS = sorted(
    ({telco.TARGET, telco.ID_COL} | set(telco.FEATURES)) - GENERIC
)

TELCO_CATEGORIES = sorted(
    {
        value
        for values in telco.CATEGORY_VALUES.values()
        for value in values
        # "Yes"/"No"/"DSL" and friends are too generic to be evidence.
        if " " in value or "-" in value
    }
)


def test_the_platform_has_modules_to_guard():
    """A guard that scans nothing passes forever."""
    assert PLATFORM_MODULES
    assert any(p.name == "reader.py" for p in PLATFORM_MODULES)


@pytest.mark.parametrize("path", PLATFORM_MODULES, ids=lambda p: p.name)
def test_no_telco_column_names_outside_the_reference_package(path):
    source = path.read_text(encoding="utf-8")
    found = [
        name
        for name in TELCO_IDENTIFIERS
        if re.search(rf"\b{re.escape(name)}\b", source)
    ]
    assert not found, (
        f"{path.relative_to(SRC.parent)} names Telco columns {found}. "
        "Platform code takes its columns from the confirmed schema mapping (I11); "
        "dataset-specific code belongs in churnkit/reference."
    )


@pytest.mark.parametrize("path", PLATFORM_MODULES, ids=lambda p: p.name)
def test_no_telco_category_values_outside_the_reference_package(path):
    source = path.read_text(encoding="utf-8")
    found = [value for value in TELCO_CATEGORIES if value in source]
    assert not found, (
        f"{path.relative_to(SRC.parent)} hardcodes Telco categories {found}"
    )


@pytest.mark.parametrize("path", PLATFORM_MODULES, ids=lambda p: p.name)
def test_no_dataset_name_outside_the_reference_package(path):
    assert "telco" not in path.read_text(encoding="utf-8").lower()


BANNED_IN_INGEST = [
    ".dropna(",
    ".fillna(",
    'errors="coerce"',
    "errors='coerce'",
    'on_bad_lines="skip"',
    "on_bad_lines='skip'",
]


@pytest.mark.parametrize(
    "path",
    sorted((SRC / "ingest").rglob("*.py")),
    ids=lambda p: p.name,
)
def test_ingest_never_drops_or_coerces_silently(path):
    source = path.read_text(encoding="utf-8")
    # Prose about the ban is fine; calling the thing is not.
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    found = [token for token in BANNED_IN_INGEST if token in code]
    assert not found, (
        f"{path.name} uses {found}. Ingest counts and reports what it cannot "
        "parse; it never quietly replaces or discards it."
    )


# ── The portal must not become a second inference path ────────────────────────

APP = Path(__file__).resolve().parents[1] / "app"

# What the UI is allowed to import from churnkit: presentation data, nothing that
# can load a model. The UI container is installed from the `ui` extra, which
# omits scikit-learn, xgboost, shap and mlflow entirely, so an import outside
# this list would also fail at runtime there — this test makes it fail in CI
# first, with an explanation.
UI_ALLOWED_IMPORTS = ("churnkit.reference.labels", "churnkit.reference.schema")

MODEL_LIBRARIES = ("sklearn", "mlflow", "xgboost", "shap", "fastapi", "uvicorn")


def _churnkit_imports(source: str) -> list[str]:
    import ast

    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("churnkit"):
                modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names if a.name.startswith("churnkit"))
    return modules


@pytest.mark.parametrize("path", sorted(APP.rglob("*.py")), ids=lambda p: p.name)
def test_the_ui_imports_only_presentation_data(path):
    imported = _churnkit_imports(path.read_text(encoding="utf-8"))
    disallowed = [m for m in imported if m not in UI_ALLOWED_IMPORTS]
    assert not disallowed, (
        f"{path.name} imports {disallowed}. The portal reaches a prediction over "
        "HTTP only; importing model code would create a second inference path "
        "that can drift from the one being served."
    )


@pytest.mark.parametrize("module", UI_ALLOWED_IMPORTS)
def test_modules_the_ui_imports_stay_free_of_model_libraries(module):
    path = SRC.parent / (module.replace(".", "/") + ".py")
    source = path.read_text(encoding="utf-8")
    found = [lib for lib in MODEL_LIBRARIES if f"import {lib}" in source]
    assert not found, (
        f"{module} imports {found}, which the UI environment does not install"
    )


# ── The portal and the API have to agree on the /reload header ────────────────
#
# The portal may not import churnkit.config (the guard above, and the `ui` extra
# it is installed from), so the header name is written out literally in the app.
# That duplication is the price of the import boundary; this test is what stops
# it drifting into a reload button that silently 403s.


def test_the_portal_sends_the_header_the_api_demands():
    from churnkit.config import ADMIN_HEADER

    source = (APP / "streamlit_app.py").read_text(encoding="utf-8")
    assert f'"{ADMIN_HEADER}"' in source, (
        f"the portal must send {ADMIN_HEADER} on /reload; the API refuses the "
        "request without it and the reload button would fail with a 403"
    )


# ── I1 — no random splitting anywhere in the platform ─────────────────────────
#
# CLAUDE.md bans random splits on data with a time dimension, and S5 asks for
# this guard by name. The reasoning is that a written instruction is forgotten
# between sessions while a failing build is not: the single most likely wrong
# line anyone adds to this repository is `train_test_split(X, y,
# random_state=42)`, because every churn tutorial ever written contains it.

RANDOM_SPLIT_TOKENS = [
    "train_test_split",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "shuffle=True",
]

# Modules permitted to use a random split, with the reason. Empty on purpose.
# Adding an entry is a deliberate act that shows up in review; if a genuine case
# arises, it belongs here with a comment saying why the data has no time
# dimension.
RANDOM_SPLIT_ALLOWED: dict[str, str] = {
    # The Telco reference dataset genuinely has no time column — churnkit's own
    # schema inference reports timestamp=None for it — and I1 permits a random
    # split in that case provided the metrics are marked optimistic. This entry
    # exists so the exception is visible rather than silent, and it disappears
    # with the module when the platform replaces it.
    "train.py": "reference implementation; its dataset has no time column (I1)",
}


def _executable_source(path) -> str:
    """Source with comments and string literals stripped.

    Prose about a banned construct is fine; calling it is not. Comparing raw
    text cannot tell the two apart, so a module explaining why it never calls
    train_test_split would fail a guard against train_test_split.
    """
    import ast

    class _Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

    return ast.unparse(_Blank().visit(ast.parse(path.read_text(encoding="utf-8"))))


@pytest.mark.parametrize(
    "path", sorted(SRC.rglob("*.py")), ids=lambda p: p.name
)
def test_no_random_splitting_in_the_platform(path):
    if path.name in RANDOM_SPLIT_ALLOWED:
        pytest.skip(RANDOM_SPLIT_ALLOWED[path.name])
    code = _executable_source(path)
    found = [token for token in RANDOM_SPLIT_TOKENS if token in code]
    assert not found, (
        f"{path.name} uses {found}. Churn is temporal: a random split trains on "
        "the future and evaluates on the past, which overstates every metric it "
        "touches (I1). Use churnkit.validation.temporal instead."
    )


def test_the_random_split_guard_scans_the_training_code():
    """A guard that scans nothing passes forever."""
    scanned = {p.name for p in SRC.rglob("*.py")}
    assert "train.py" in scanned
    assert "temporal.py" in scanned
