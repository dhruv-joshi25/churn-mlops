"""Turn an event log into per-customer features, without looking forward.

A snapshot table says a customer makes 15 logins a month. An event log says they
fell from 15 to 3, and the fall is the signal — this is where accuracy actually
comes from, not from a larger model.

Getting it requires reaching backwards through history, and reaching one day too
far *forwards* produces a feature that predicts the future because it already
contains it. That failure is silent: it inflates every validation metric,
survives into the model artifact undetected, and only shows up as a model that
worked in testing and does nothing in production.

So the window is strict. Events are used only where ``timestamp <
prediction_date``; an event dated exactly on the prediction date is not usable
when scoring that date, because at the moment of scoring it has not finished
happening. Every feature also records the latest event timestamp that fed it,
and :func:`~churnkit.validation.temporal.assert_no_time_travel` is called on
those timestamps inside the build — not merely in a test, where a future
refactor would never reach it (I2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from churnkit.validation.temporal import assert_no_time_travel

__all__ = ["FEATURE_GROUPS", "FeatureMatrix", "build_features"]

DEFAULT_WINDOWS: tuple[int, ...] = (7, 30, 90)

# Selectable so snapshot-only can be benchmarked against snapshot+events (S6).
# Passing a subset is how that comparison gets made without a second code path
# that could drift from this one.
FEATURE_GROUPS: tuple[str, ...] = (
    "counts",
    "deltas",
    "recency",
    "frequency",
    "monetary",
    "trend",
    "gaps",
    "flags",
)

# A month-on-month fall of this much is the threshold flag S6 asks for. It is a
# flag rather than a filter: the model decides what it is worth.
USAGE_DROP_RATIO = 0.7

TREND_WINDOW_DAYS = 90


@dataclass(frozen=True)
class FeatureMatrix:
    """Features plus the provenance that proves they are point-in-time.

    ``source_dates`` carries, for every feature of every row, the latest event
    timestamp that contributed to it. It is what makes the I2 guarantee
    checkable by something other than reading the code.
    """

    features: pd.DataFrame
    source_dates: pd.DataFrame
    windows: tuple[int, ...]
    groups: tuple[str, ...]
    n_events_used: int
    n_events_excluded_as_future: int
    warnings: tuple[str, ...] = ()


def _slope(days_before: np.ndarray[Any, Any]) -> float:
    """Least-squares trend of daily activity, over the span actually observed.

    Negative means the customer is winding down. Returns 0.0 rather than NaN
    where there is nothing to fit, because "no trend" is the honest reading of
    one event and a NaN would silently drop the row from most estimators.

    The fit runs from the customer's earliest event inside the window up to the
    prediction date, **not** across the full ninety days. Padding the empty time
    before someone's first event with zeros makes a leaving customer look
    steady: their real decline sits in the middle of the series, with flat zero
    on both sides, and a straight line through that is horizontal. It also
    quietly penalises anyone who has not been a customer for the whole window,
    which is every recent signup.
    """
    if len(days_before) < 2:
        return 0.0
    span = int(np.ceil(days_before.max())) + 1
    if span < 2:
        return 0.0
    counts = np.bincount(days_before.astype(int), minlength=span)[:span]
    if counts.sum() == 0 or np.all(counts == counts[0]):
        return 0.0
    # x counts days *before* the prediction date, so a positive fit means more
    # activity further back — i.e. decline. Negate so the sign reads naturally.
    return float(-np.polyfit(np.arange(span), counts, 1)[0])


def _longest_gap(
    timestamps: pd.Series, start: pd.Timestamp, as_of: pd.Timestamp
) -> float:
    """Longest stretch with no events inside the window, in days.

    The window edges count: a customer silent for the first two months of a
    ninety-day window has a sixty-day gap whether or not an event bookends it.
    """
    points = [start, *sorted(timestamps), as_of]
    spans = (
        (b - a).total_seconds()
        for a, b in zip(points, points[1:], strict=False)
    )
    return float(max(spans) / 86400)


def _clean_name(value: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_").lower()


def _features_for(
    history: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    timestamp_column: str,
    type_column: str | None,
    value_column: str | None,
    windows: Sequence[int],
    groups: Sequence[str],
    event_types: Sequence[object],
) -> tuple[dict[str, float], pd.Timestamp | None]:
    """Every feature for one customer at one date, from ``history`` alone.

    ``history`` has already been filtered to events strictly before ``as_of``.
    Nothing in here re-reads the full log, so there is no path by which a later
    event can enter a calculation.
    """
    times = history[timestamp_column]
    out: dict[str, float] = {}
    latest = times.max() if len(times) else None

    def within(days: int, offset: int = 0) -> pd.DataFrame:
        upper = as_of - pd.Timedelta(days=days * offset)
        lower = as_of - pd.Timedelta(days=days * (offset + 1))
        return history[(times >= lower) & (times < upper)]

    for window in windows:
        current = within(window)
        if "counts" in groups:
            out[f"events_{window}d"] = float(len(current))
            for event_type in event_types:
                subset = current[current[type_column] == event_type]
                out[f"events_{_clean_name(event_type)}_{window}d"] = float(len(subset))

        if "deltas" in groups:
            previous = within(window, offset=1)
            out[f"events_{window}d_delta"] = float(len(current) - len(previous))
            out[f"events_{window}d_ratio"] = (
                float(len(current) / len(previous)) if len(previous) else np.nan
            )

        if "monetary" in groups and value_column is not None:
            amounts = pd.to_numeric(current[value_column])
            has_any = len(amounts) > 0
            out[f"value_sum_{window}d"] = float(amounts.sum()) if has_any else 0.0
            out[f"value_mean_{window}d"] = float(amounts.mean()) if has_any else 0.0

    if "recency" in groups:
        out["days_since_last_event"] = (
            float((as_of - latest).total_seconds() / 86400)
            if latest is not None
            else np.nan
        )
        for event_type in event_types:
            subset = history[history[type_column] == event_type]
            last = subset[timestamp_column].max() if len(subset) else None
            out[f"days_since_last_{_clean_name(event_type)}"] = (
                float((as_of - last).total_seconds() / 86400)
                if last is not None
                else np.nan
            )

    if "frequency" in groups:
        active_days = times.dt.normalize().nunique() if len(times) else 0
        out["events_per_active_day"] = (
            float(len(times) / active_days) if active_days else 0.0
        )

    if "trend" in groups:
        recent = times[times >= as_of - pd.Timedelta(days=TREND_WINDOW_DAYS)]
        days_before = ((as_of - recent).dt.total_seconds() / 86400).to_numpy()
        out[f"engagement_slope_{TREND_WINDOW_DAYS}d"] = _slope(days_before)

    if "gaps" in groups:
        for window in windows:
            start = as_of - pd.Timedelta(days=window)
            current = within(window)
            out[f"longest_gap_{window}d"] = _longest_gap(
                current[timestamp_column], start, as_of
            )

    if "flags" in groups:
        this_month = len(within(30))
        last_month = len(within(30, offset=1))
        out["usage_down_30pct_mom"] = float(
            last_month > 0 and this_month < USAGE_DROP_RATIO * last_month
        )

    return out, latest


def build_features(
    events: pd.DataFrame,
    *,
    customer_column: str,
    timestamp_column: str,
    prediction_dates: pd.Timestamp | Mapping[object, pd.Timestamp] | pd.Series,
    type_column: str | None = None,
    value_column: str | None = None,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    groups: Sequence[str] = FEATURE_GROUPS,
) -> FeatureMatrix:
    """Point-in-time features per customer per prediction date.

    ``prediction_dates`` is either one date for every customer, or a mapping
    from customer to their own date — the second form is what walk-forward folds
    need, where each customer is scored as of the fold they fall in.

    ``groups`` selects which families of feature to compute, which is the hook
    for benchmarking a snapshot-only model against one with event features. It
    is a filter on this single implementation rather than a second one, so the
    comparison cannot be confounded by two code paths drifting apart.
    """
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown feature group(s) {unknown}. Available groups: "
            f"{', '.join(FEATURE_GROUPS)}"
        )
    for column in (customer_column, timestamp_column):
        if column not in events.columns:
            known = ", ".join(map(str, events.columns))
            raise KeyError(
                f"no column named {column!r} in the event log. Present: {known}"
            )

    frame = events.copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column])

    if isinstance(prediction_dates, pd.Timestamp | str):
        as_of_by_customer = pd.Series(
            pd.Timestamp(prediction_dates), index=frame[customer_column].unique()
        )
    else:
        as_of_by_customer = pd.Series(prediction_dates)
    as_of_by_customer = pd.to_datetime(as_of_by_customer)

    event_types: tuple[object, ...] = ()
    if type_column is not None and type_column in frame.columns:
        event_types = tuple(sorted(frame[type_column].dropna().unique(), key=str))

    rows: dict[tuple[object, pd.Timestamp], dict[str, float]] = {}
    sources: dict[tuple[object, pd.Timestamp], pd.Timestamp | None] = {}
    used = 0
    excluded_future = 0

    by_customer = dict(list(frame.groupby(customer_column, sort=False)))

    for customer, as_of in as_of_by_customer.items():
        history = by_customer.get(customer)
        if history is None:
            history = frame.iloc[0:0]
        # THE line this module exists for. Strictly before, never on or after.
        past = history[history[timestamp_column] < as_of]
        used += len(past)
        excluded_future += len(history) - len(past)

        computed, latest = _features_for(
            past,
            as_of,
            timestamp_column=timestamp_column,
            type_column=type_column,
            value_column=value_column,
            windows=windows,
            groups=groups,
            event_types=event_types,
        )
        rows[(customer, as_of)] = computed
        sources[(customer, as_of)] = latest

    index = pd.MultiIndex.from_tuples(rows.keys(), names=[customer_column, "as_of"])
    features = pd.DataFrame(list(rows.values()), index=index)

    # Provenance: the latest event that fed any feature on this row. Every
    # feature shares it because every feature was computed from `past`, which is
    # bounded by that timestamp.
    latest_per_row = pd.Series(list(sources.values()), index=index)
    source_dates = pd.DataFrame(
        dict.fromkeys(features.columns, latest_per_row), index=index
    )

    # Inside the generation path, so a refactor that breaks the windowing fails
    # here rather than shipping a model that looks excellent (S6, I2).
    assert_no_time_travel(
        source_dates, pd.Series(index.get_level_values("as_of"))
    )

    warnings: list[str] = []
    if excluded_future:
        warnings.append(
            f"{excluded_future} event(s) fell on or after their customer's "
            "prediction date and were excluded. That is the intended behaviour, "
            "not a data problem: an event is only usable strictly before the "
            "moment being scored"
        )

    return FeatureMatrix(
        features=features,
        source_dates=source_dates,
        windows=tuple(windows),
        groups=tuple(groups),
        n_events_used=used,
        n_events_excluded_as_future=excluded_future,
        warnings=tuple(warnings),
    )
