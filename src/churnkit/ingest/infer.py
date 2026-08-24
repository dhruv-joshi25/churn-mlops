"""Propose what each column is, and refuse to let a guess reach training.

Two jobs, and the second is the reason the first exists.

**Role inference** reads a :class:`~churnkit.ingest.reader.ParseResult` and says
what every column looks like — an identifier, a date, a category, the churn
target. Every answer carries a confidence and a sentence of reasoning, because a
proposal an operator cannot argue with is a proposal they will rubber-stamp.

**Leakage detection** then asks whether any column knows the answer. A churn
model trained on a `cancellation_date` scores 0.99 AUC in validation and is
worthless in production, and this is the single most common way the tutorials
this project exists to improve on get it wrong.

Nothing here decides anything (I7). :func:`infer_schema` returns a
:class:`SchemaProposal`, and a proposal cannot be turned into a training mapping
without :meth:`SchemaProposal.confirm` — which, where a blocking leakage finding
exists, requires the operator to type a phrase rather than pass a flag (I5).

On thresholds: over-flagging is recoverable, because the operator can override
it. Under-flagging is not, because nothing downstream will catch it. So where
the evidence is ambiguous this module blocks and explains itself. What stops
that reasoning collapsing into "flag everything" is
``tests/fixtures/leaky/06_clean_baseline.csv``, which must produce zero blocking
findings — a detector that cries wolf teaches operators to override the one
finding that mattered. ADR 0005 records the thresholds and the trade.

No column name, category or business rule from any particular dataset appears
here; everything is inferred from the file in front of it (I11).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from churnkit.ingest.reader import ParseResult

__all__ = [
    "ColumnRole",
    "ConfirmedMapping",
    "LeakageFinding",
    "LeakageOverrideRequired",
    "SchemaProposal",
    "SkippedCheck",
    "TargetCandidate",
    "UnconfirmedProposalError",
    "infer_schema",
]

# ── Roles ─────────────────────────────────────────────────────────────────────

IDENTIFIER = "identifier"
NUMERIC = "numeric"
CATEGORICAL_LOW = "categorical_low"
CATEGORICAL_HIGH = "categorical_high"
DATETIME = "datetime"
FREE_TEXT = "free_text"
CONSTANT = "constant"
TARGET_CANDIDATE = "target_candidate"

# A category is "low cardinality" if a human could read the whole list. Above
# this a one-hot encoding stops being sensible and the column needs a different
# treatment, which is a decision for the operator rather than a default.
MAX_LOW_CARDINALITY = 20

# Free text is long and rarely repeats. Both conditions are required: a short
# near-unique string is a product code, and a long repeated one is a category.
FREE_TEXT_MIN_MEAN_LENGTH = 40

# ── Threshold tiers (ADR 0005) ────────────────────────────────────────────────

# A single column separating the classes this well is not a feature, it is the
# answer wearing a different name. Real churn features top out far below this —
# tenure and contract type reach roughly 0.65-0.75 alone.
AUC_BLOCKING = 0.95

# Above this, or at exact separation, the finding blocks whatever the sample
# size: no amount of small-n luck produces a flawless ordering of a real feature.
AUC_CERTAIN = 0.99

# Whether the value is *missing* should not tell you the outcome. Where it does,
# the column exists because of the outcome.
NULL_MASK_BLOCKING = 0.90
NULL_MASK_CERTAIN = 0.99

# Below this many rows in the smaller class, a borderline statistic is luck. It
# still gets reported — as a warning, so the operator sees it without the run
# being halted over twelve rows.
MIN_CLASS_FOR_CONFIDENCE = 50

# Name patterns that are specific enough to block on the name alone. Each of
# these describes an event that happens *because* the customer left.
NAME_PATTERNS_STRONG = re.compile(
    r"cancel|churn|terminat|exit|refund|closed|close_date|lost|attrit|deactivat",
    re.IGNORECASE,
)

# Patterns worth surfacing but not worth halting for. "status" matches
# marital_status and employment_status; "reason" matches reason_for_signup;
# an end_date may be a contract end known at signup. These warn, and escalate to
# blocking only when a statistical rule fires on the same column.
#
# "last_" is deliberately NOT here. last_login, last_payment and last_seen are
# the most valuable legitimate churn features there are, and flagging them is
# precisely the cry-wolf failure 06_clean_baseline.csv exists to prevent.
NAME_PATTERNS_WEAK = re.compile(r"status|reason|end_date|final_", re.IGNORECASE)

TARGET_NAME_PATTERN = re.compile(
    r"churn|left|leave|lapsed|attrit|cancel|stopped|gone|renew|exit|closed|lost"
    r"|inactive|target|label",
    re.IGNORECASE,
)

ID_NAME_STRONG = re.compile(
    r"(^|_)(id|uuid|guid|key)$|^(id|uuid|guid)|_id$|customer|account|member|client"
    r"|subscriber|user",
    re.IGNORECASE,
)

ID_NAME_WEAK = re.compile(r"ref|reference|code|number|(^|_)no$", re.IGNORECASE)

TIMESTAMP_NAME_PATTERN = re.compile(
    r"signup|sign_up|signed|created|opened|start|joined|registered|as_of|snapshot",
    re.IGNORECASE,
)

BINARY_TRUE_FALSE = ({0, 1}, {"yes", "no"}, {"true", "false"}, {"y", "n"})

OVERRIDE_PHRASE = "I ACCEPT THE LEAKAGE RISK"


# ── Result types ──────────────────────────────────────────────────────────────


class UnconfirmedProposalError(RuntimeError):
    """Raised when an unconfirmed proposal is asked for a training mapping (I7)."""


class LeakageOverrideRequired(RuntimeError):
    """Raised when confirmation is attempted over a blocking finding (I5)."""


@dataclass(frozen=True)
class ColumnRole:
    name: str
    role: str
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class TargetCandidate:
    name: str
    confidence: float
    positive_rate: float
    reasoning: str


@dataclass(frozen=True)
class LeakageFinding:
    column: str
    rule: str
    severity: str  # "blocking" | "warning"
    evidence: str
    detail: str


@dataclass(frozen=True)
class SkippedCheck:
    """A check that did not run. Never reported as a pass — see the 04 manifest."""

    rule: str
    reason: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class ConfirmedMapping:
    """The only thing training accepts. Reachable only through confirm()."""

    target: str
    id_column: str | None
    timestamp_column: str | None
    roles: Mapping[str, ColumnRole]
    excluded: Mapping[str, str]
    overridden_findings: tuple[LeakageFinding, ...] = ()
    confirmed_at: datetime | None = None


@dataclass(frozen=True)
class SchemaProposal:
    roles: Mapping[str, ColumnRole]
    target_candidates: tuple[TargetCandidate, ...]
    id_column: str | None
    timestamp_column: str | None
    excluded: Mapping[str, str]
    leakage: tuple[LeakageFinding, ...]
    skipped_checks: tuple[SkippedCheck, ...]
    warnings: tuple[str, ...] = field(default=())

    @property
    def target(self) -> str | None:
        return self.target_candidates[0].name if self.target_candidates else None

    @property
    def blocking_findings(self) -> tuple[LeakageFinding, ...]:
        return tuple(f for f in self.leakage if f.severity == "blocking")

    @property
    def requires_typed_override(self) -> bool:
        return bool(self.blocking_findings)

    @property
    def override_phrase(self) -> str:
        return OVERRIDE_PHRASE

    def to_mapping(self) -> ConfirmedMapping:
        """Always raises. Exists so the I7 boundary is a type error, not a habit.

        There is deliberately no argument that makes this work. Reaching a
        training mapping goes through :meth:`confirm`, which cannot be called
        without naming a target — which is to say, without a human having read
        the proposal.
        """
        raise UnconfirmedProposalError(
            "a SchemaProposal is a proposal, not a mapping (I7). Call confirm() "
            "with the target, id and timestamp columns a human has reviewed. "
            + (
                f"{len(self.blocking_findings)} blocking leakage finding(s) must "
                f"be overridden by typing {OVERRIDE_PHRASE!r}."
                if self.requires_typed_override
                else "This proposal has no blocking findings."
            )
        )

    def confirm(
        self,
        *,
        target: str,
        id_column: str | None = None,
        timestamp_column: str | None = None,
        override_phrase: str | None = None,
        now: datetime | None = None,
    ) -> ConfirmedMapping:
        """Turn a reviewed proposal into the mapping training accepts.

        ``override_phrase`` must be typed out in full where blocking findings
        exist. A boolean flag would be a single character in a config file; a
        phrase has to be written by someone who read what they are overriding,
        and it is recorded on the mapping so the decision travels with the run.
        """
        if target not in self.roles:
            known = ", ".join(sorted(self.roles))
            raise ValueError(
                f"cannot confirm target {target!r}: no such column. Columns "
                f"read from this file: {known}"
            )

        blocking = self.blocking_findings
        if blocking and override_phrase != OVERRIDE_PHRASE:
            named = ", ".join(f"{f.column} ({f.rule})" for f in blocking)
            raise LeakageOverrideRequired(
                f"{len(blocking)} blocking leakage finding(s) — {named}. Training "
                f"on these produces a model that scores well and predicts nothing. "
                f"To proceed anyway, pass override_phrase={OVERRIDE_PHRASE!r} "
                f"exactly; the override is recorded on the mapping."
            )

        return ConfirmedMapping(
            target=target,
            id_column=id_column,
            timestamp_column=timestamp_column,
            roles=self.roles,
            excluded=self.excluded,
            overridden_findings=blocking,
            confirmed_at=now,
        )


# ── Statistics ────────────────────────────────────────────────────────────────


def _positive_class_mask(values: pd.Series) -> np.ndarray[Any, Any]:
    """Which rows are the positive class, without coercing anything.

    The churn event is the larger of the two values a binary column holds — 1
    over 0, "yes" over "no", True over False. Comparing against it directly
    avoids a numeric conversion that would silently turn an unexpected spelling
    into a missing value, which is the behaviour the ingest layer bans.
    """
    present = values[values.notna()]
    distinct = sorted(present.unique(), key=str)
    if not distinct:
        return np.zeros(len(values), dtype=int)
    mask: np.ndarray[Any, Any] = (values == distinct[-1]).to_numpy().astype(int)
    return mask


def _rank_auc(values: pd.Series, y: np.ndarray[Any, Any]) -> float | None:
    """Direction-free AUC by ranks. None when it cannot be computed honestly.

    Rank-based rather than threshold-based so ties are handled correctly, and
    direction-free because a column that predicts churn perfectly *backwards* is
    exactly as leaky as one that predicts it forwards.
    """
    mask = values.notna().to_numpy()
    x, yy = values.to_numpy()[mask], y[mask]
    n_pos, n_neg = int((yy == 1).sum()), int((yy == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(x).rank().to_numpy()
    auc = (ranks[yy == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))


def _null_mask_correlation(values: pd.Series, y: np.ndarray[Any, Any]) -> float | None:
    """How much the *absence* of a value tells you about the outcome."""
    mask = values.isna().to_numpy().astype(float)
    if mask.std() == 0 or y.std() == 0:
        return None
    return float(abs(np.corrcoef(mask, y)[0, 1]))


def _is_constant_within_class(values: pd.Series, y: np.ndarray[Any, Any]) -> bool:
    """One value per class — perfect separation without needing to be numeric.

    Requires the column to vary overall. A column with a single value throughout
    is trivially constant inside every class, and is a useless column rather
    than a leak; treating it as one was a real false positive this rule had
    before ``07_roles.csv`` caught it.
    """
    if values[values.notna()].nunique() <= 1:
        return False
    seen = set()
    for klass in (0, 1):
        column = values[y == klass]
        distinct = column[column.notna()].unique()
        if len(distinct) != 1:
            return False
        seen.add(distinct[0])
    return len(seen) > 1


def _tier(certain: bool, borderline: bool, min_class: int) -> str:
    """Blocking unless the only evidence is a borderline statistic on few rows."""
    if certain:
        return "blocking"
    if borderline and min_class < MIN_CLASS_FOR_CONFIDENCE:
        return "warning"
    return "blocking"


# ── Role inference ────────────────────────────────────────────────────────────


def _looks_binary(values: pd.Series) -> bool:
    distinct = values[values.notna()].unique()
    if len(distinct) != 2:
        return False
    as_set = {str(v).strip().lower() for v in distinct}
    numeric = set()
    for v in distinct:
        try:
            numeric.add(int(float(v)))
        except (TypeError, ValueError):
            numeric = set()
            break
    return any(as_set == p or numeric == p for p in BINARY_TRUE_FALSE)


def _role_for(
    name: str, values: pd.Series, kind: str, n_rows: int
) -> tuple[str, float, str]:
    distinct = int(values[values.notna()].nunique())

    if distinct <= 1:
        return CONSTANT, 0.99, (
            f"one value across all {n_rows} rows, so it carries no information"
        )

    if _looks_binary(values) and (
        kind == "numeric" or TARGET_NAME_PATTERN.search(name)
    ):
        hit = TARGET_NAME_PATTERN.search(name)
        conf = 0.9 if hit else 0.55
        why = "two values" + (
            f" and the name matches {hit.group(0)!r}" if hit else ", but the name "
            "suggests nothing about churn"
        )
        return TARGET_CANDIDATE, conf, why

    if kind == "datetime":
        return DATETIME, 0.95, "parsed as dates by the reader"

    if kind == "numeric":
        return NUMERIC, 0.9, f"{distinct} distinct numeric values"

    lengths = values[values.notna()].astype(str).str.len()
    mean_length = float(lengths.mean()) if len(lengths) else 0.0
    unique_ratio = distinct / max(n_rows, 1)

    if mean_length >= FREE_TEXT_MIN_MEAN_LENGTH and unique_ratio > 0.5:
        return FREE_TEXT, 0.85, (
            f"averages {mean_length:.0f} characters and is {unique_ratio:.0%} "
            "unique, so it reads as prose rather than a category"
        )

    if distinct <= min(MAX_LOW_CARDINALITY, max(2, int(n_rows * 0.5))):
        return CATEGORICAL_LOW, 0.9, f"{distinct} distinct values"

    return CATEGORICAL_HIGH, 0.75, (
        f"{distinct} distinct values across {n_rows} rows ({unique_ratio:.0%} "
        "unique), too many to one-hot without a decision from you"
    )


def _identifier_rank(
    name: str, position: int, unique_ratio: float
) -> tuple[int, int] | None:
    """Rank a fully-unique column's claim to be *the* identifier.

    Several columns can be unique — a row key and a product reference both are.
    Only one is the customer identifier, so the claims are ranked and the rest
    keep the role their content earns them.
    """
    if unique_ratio < 1.0:
        return None
    if ID_NAME_STRONG.search(name):
        return (0, position)
    if ID_NAME_WEAK.search(name):
        return (1, position)
    return None


# ── Leakage ───────────────────────────────────────────────────────────────────


def _name_findings(name: str, statistical: bool) -> list[LeakageFinding]:
    strong = NAME_PATTERNS_STRONG.search(name)
    if strong:
        return [
            LeakageFinding(
                column=name,
                rule="name_pattern",
                severity="blocking",
                evidence=f"name contains {strong.group(0)!r}",
                detail=(
                    "columns named for the ending of a relationship are recorded "
                    "because the customer left, so their value is unknown at the "
                    "moment a prediction would be made"
                ),
            )
        ]
    weak = NAME_PATTERNS_WEAK.search(name)
    if weak:
        return [
            LeakageFinding(
                column=name,
                rule="name_pattern",
                severity="blocking" if statistical else "warning",
                evidence=f"name contains {weak.group(0)!r}",
                detail=(
                    "this pattern also matches ordinary features such as "
                    "marital_status or reason_for_signup, so on its own it is "
                    "worth a look rather than a halt"
                    if not statistical
                    else "raised to blocking because a statistical rule also "
                    "fired on this column"
                ),
            )
        ]
    return []


def _statistical_findings(
    name: str,
    values: pd.Series,
    y: np.ndarray[Any, Any],
    kind: str,
    min_class: int
) -> list[LeakageFinding]:
    out: list[LeakageFinding] = []

    if kind == "numeric":
        auc = _rank_auc(values, y)
        if auc is not None and auc >= AUC_BLOCKING:
            out.append(
                LeakageFinding(
                    column=name,
                    rule="single_column_auc",
                    severity=_tier(auc >= AUC_CERTAIN, True, min_class),
                    evidence=f"AUC {auc:.3f} against the target on its own",
                    detail=(
                        "no genuine churn feature separates the classes this "
                        "cleanly by itself; a column that does is usually the "
                        "outcome restated"
                    ),
                )
            )

    corr = _null_mask_correlation(values, y)
    if corr is not None and corr >= NULL_MASK_BLOCKING:
        out.append(
            LeakageFinding(
                column=name,
                rule="null_mask",
                severity=_tier(corr >= NULL_MASK_CERTAIN, True, min_class),
                evidence=f"missingness correlates {corr:.3f} with the target",
                detail=(
                    "whether this column has a value tells you the outcome, so "
                    "the column exists as a consequence of the outcome"
                ),
            )
        )

    if _is_constant_within_class(values, y):
        out.append(
            LeakageFinding(
                column=name,
                rule="constant_within_class",
                severity="blocking",
                evidence="one distinct value within each target class",
                detail=(
                    "the column separates the classes perfectly without being "
                    "numeric, which the AUC check cannot see"
                ),
            )
        )

    return out


# ── Entry point ───────────────────────────────────────────────────────────────


def infer_schema(
    result: ParseResult,
    *,
    target: str | None = None,
    observation_end: date | None = None,
) -> SchemaProposal:
    """Propose roles and report leakage. Decides nothing (I7).

    ``target`` names the churn column when the operator already knows it;
    without it the ranked candidates are inferred and the first is used for the
    leakage checks, which is a guess and is reported as one.

    ``observation_end`` is the last date features may see. Without it the
    post-window check cannot run, and is reported as skipped rather than passed.
    """
    frame = result.frame
    n_rows = len(frame)
    kinds = {name: stats.kind for name, stats in result.columns.items()}

    roles: dict[str, ColumnRole] = {}
    id_claims: list[tuple[tuple[int, int], str]] = []

    for position, name in enumerate(frame.columns):
        values = frame[name]
        kind = kinds.get(name, "text")
        role, confidence, why = _role_for(name, values, kind, n_rows)
        roles[name] = ColumnRole(name, role, confidence, why)

        unique_ratio = int(values[values.notna()].nunique()) / max(n_rows, 1)
        if role in (CATEGORICAL_HIGH, CATEGORICAL_LOW, FREE_TEXT, NUMERIC):
            claim = _identifier_rank(name, position, unique_ratio)
            if claim is not None:
                id_claims.append((claim, name))

    id_column = None
    if id_claims:
        id_column = min(id_claims)[0:2][1]
        roles[id_column] = ColumnRole(
            id_column,
            IDENTIFIER,
            0.9,
            f"unique across all {n_rows} rows and named like an identifier",
        )

    # Target candidates, ranked. A name that speaks to churn beats one that
    # merely happens to be binary.
    candidates: list[TargetCandidate] = []
    for name, column_role in roles.items():
        if column_role.role != TARGET_CANDIDATE:
            continue
        rate = float(_positive_class_mask(frame[name]).mean())
        hit = TARGET_NAME_PATTERN.search(name)
        candidates.append(
            TargetCandidate(
                name=name,
                confidence=0.9 if hit else 0.5,
                positive_rate=rate,
                reasoning=(
                    f"binary column whose name matches {hit.group(0)!r}"
                    if hit
                    else "binary column, but nothing in the name refers to churn"
                ),
            )
        )
    candidates.sort(key=lambda c: (-c.confidence, c.name))

    if target is not None:
        candidates.sort(key=lambda c: (c.name != target, -c.confidence))

    chosen = target or (candidates[0].name if candidates else None)

    warnings: list[str] = []
    if chosen is None:
        warnings.append(
            "no target column could be identified: no column is binary, so the "
            "churn definition has to be supplied before anything can be trained"
        )

    # Timestamp. Leaky datetimes are not eligible — proposing a cancellation
    # date as the observation timestamp would be the leak choosing itself.
    datetimes = [n for n, r in roles.items() if r.role == DATETIME]
    leak_names = {
        f.column
        for f in _prescan_names(datetimes)
    }
    eligible = [n for n in datetimes if n not in leak_names]
    timestamp_column = None
    if eligible:
        preferred = [n for n in eligible if TIMESTAMP_NAME_PATTERN.search(n)]
        timestamp_column = (preferred or eligible)[0]
    if timestamp_column is None:
        warnings.append(
            "no timestamp column was found, so temporal validation cannot be "
            "performed and any metric produced from this data is optimistic (I1)"
        )

    # Leakage, against the chosen target.
    findings: list[LeakageFinding] = []
    skipped: list[SkippedCheck] = []

    if chosen is None:
        skipped.append(
            SkippedCheck(
                rule="all",
                reason="no target column identified, so nothing can be tested "
                "against it",
                columns=tuple(frame.columns),
            )
        )
    else:
        y = _positive_class_mask(frame[chosen])
        min_class = int(min((y == 1).sum(), (y == 0).sum()))

        if min_class < MIN_CLASS_FOR_CONFIDENCE:
            warnings.append(
                f"the smaller target class has {min_class} rows, below the "
                f"{MIN_CLASS_FOR_CONFIDENCE} needed to trust a borderline "
                "statistic; borderline findings are reported as warnings"
            )

        for name in frame.columns:
            if name == chosen or roles[name].role in (IDENTIFIER, CONSTANT):
                continue
            stat = _statistical_findings(
                name, frame[name], y, kinds.get(name, "text"), min_class
            )
            findings.extend(stat)
            findings.extend(_name_findings(name, statistical=bool(stat)))

    # Post-window dates.
    ambiguous = set(result.ambiguous_date_columns)
    checkable = [n for n in datetimes if n not in ambiguous and n != chosen]
    if ambiguous:
        skipped.append(
            SkippedCheck(
                rule="post_window_datetime",
                reason="the reader could not decide day-first from month-first "
                "for these columns, and a misread date lands past the window "
                "spuriously; disambiguate them first",
                columns=tuple(sorted(ambiguous)),
            )
        )
    if observation_end is None:
        skipped.append(
            SkippedCheck(
                rule="post_window_datetime",
                reason="no observation_end was supplied, so there is no window "
                "to compare against. This check did not run — that is not the "
                "same as it passing",
                columns=tuple(checkable),
            )
        )
    else:
        cutoff = pd.Timestamp(observation_end)
        for name in checkable:
            column = frame[name]
            present = column[column.notna()]
            after = present[present > cutoff]
            if len(after):
                findings.append(
                    LeakageFinding(
                        column=name,
                        rule="post_window_datetime",
                        severity="blocking",
                        evidence=(
                            f"{len(after)} value(s) after {observation_end}, "
                            f"latest {after.max().date()}"
                        ),
                        detail=(
                            "a feature computed from this column would use "
                            "information that did not exist at the prediction "
                            "date (I2)"
                        ),
                    )
                )

    excluded = {
        name: (
            "identifier — unique per row, so it cannot generalise"
            if role.role == IDENTIFIER
            else "constant — one value throughout, so it carries no information"
            if role.role == CONSTANT
            else "free text — needs a decision on how to encode it before use"
        )
        for name, role in roles.items()
        if role.role in (IDENTIFIER, CONSTANT, FREE_TEXT)
    }

    return SchemaProposal(
        roles=roles,
        target_candidates=tuple(candidates),
        id_column=id_column,
        timestamp_column=timestamp_column,
        excluded=excluded,
        leakage=tuple(findings),
        skipped_checks=tuple(skipped),
        warnings=tuple(warnings),
    )


def _prescan_names(names: Sequence[str]) -> list[LeakageFinding]:
    """Name-pattern hits only, used to keep a leaky date out of the timestamp slot."""
    out: list[LeakageFinding] = []
    for name in names:
        out.extend(_name_findings(name, statistical=False))
    return [f for f in out if f.severity == "blocking"]
