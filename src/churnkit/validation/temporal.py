"""Split data by time, label it honestly, and refuse to look into the future.

This module is the project's whole argument. Every open-source churn repository
does `train_test_split(X, y, random_state=42)`, and that single line trains the
model on the future and evaluates it on the past. The metric that comes out is
not merely optimistic, it is measuring a task nobody will ever be asked to
perform. Being the project that gets this right is the point (I1).

Three things here are deliberately unlike the tutorials:

* **The cutoff is strict.** A row dated exactly on the cutoff is evaluation, not
  training. Boundary rows are the ones most likely to leak, so the boundary is
  resolved against the model rather than in its favour.

* **Censoring is not a negative class** (I3). A customer who has not churned and
  who was not observed for the full prediction horizon has an *unknown* outcome.
  Labelling them 0 teaches the model that "we stopped watching" means "stayed",
  which is how a model learns to predict its own data collection. They are
  excluded and counted.

* **Every dropped row is explained** (I10). :class:`LabelResult` carries a
  report naming each exclusion reason, its count and a sample of the identifiers
  affected, because a labelling step that quietly discards a third of the data
  produces a model nobody can account for.

There is no shuffling anywhere in this file, and no import of
``train_test_split``. A structural guard in ``tests/test_layout_guards.py``
fails the build if one appears, because a written instruction is forgotten
across sessions and a failing build is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

__all__ = [
    "ExcludedRows",
    "Fold",
    "LabelResult",
    "TemporalSplit",
    "TimeTravelError",
    "assert_no_time_travel",
    "cutoff_split",
    "label_with_censoring",
    "walk_forward_folds",
]

MAX_SAMPLE_IDS = 5


class TimeTravelError(AssertionError):
    """A feature was computed from data that postdates its prediction (I2)."""


@dataclass(frozen=True)
class TemporalSplit:
    cutoff: pd.Timestamp
    train_index: pd.Index
    eval_index: pd.Index
    n_undated: int
    date_column: str
    warnings: tuple[str, ...] = ()

    @property
    def n_train(self) -> int:
        return len(self.train_index)

    @property
    def n_eval(self) -> int:
        return len(self.eval_index)

    @property
    def reported_auc_is_temporal(self) -> bool:
        """Always True, and it exists to be asserted in a test.

        There is no mode of this object that produces a random split, so any
        metric computed from its indices is a temporal one. The property makes
        that guarantee something a test can hold onto rather than a claim in a
        docstring.
        """
        return True


@dataclass(frozen=True)
class Fold:
    index: int
    train_index: pd.Index
    eval_index: pd.Index
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp


@dataclass(frozen=True)
class ExcludedRows:
    reason: str
    n: int
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class LabelResult:
    labels: pd.Series
    censored: pd.Index
    excluded: tuple[ExcludedRows, ...]
    observation_end: pd.Timestamp
    horizon_end: pd.Timestamp
    n_input: int

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean()) if len(self.labels) else 0.0


def _dates(frame: pd.DataFrame, date_column: str) -> pd.Series:
    if date_column not in frame.columns:
        known = ", ".join(map(str, frame.columns))
        raise KeyError(
            f"no column named {date_column!r} to split on. Columns present: {known}"
        )
    values = frame[date_column]
    if not pd.api.types.is_datetime64_any_dtype(values):
        raise TypeError(
            f"column {date_column!r} holds {values.dtype}, not dates. Parse it "
            "with churnkit.ingest.read_table before splitting on it, so that "
            "unparseable values are reported rather than silently coerced"
        )
    return values


def cutoff_split(
    frame: pd.DataFrame, date_column: str, cutoff: pd.Timestamp
) -> TemporalSplit:
    """Train strictly before ``cutoff``, evaluate from ``cutoff`` onward.

    The inequality is strict on the training side on purpose. Rows sitting
    exactly on the boundary are the ones whose labels are most likely to have
    been influenced by what happens just after it, so they go to evaluation —
    the split is resolved against the model, never in its favour.
    """
    cutoff = pd.Timestamp(cutoff)
    values = _dates(frame, date_column)

    dated = values.notna()
    n_undated = int((~dated).sum())
    warnings: list[str] = []
    if n_undated:
        warnings.append(
            f"{n_undated} row(s) have no date in {date_column!r} and were "
            "excluded from both sides of the split: a row with no date cannot "
            "be placed in time, and guessing a side would put unknown data on "
            "one of them"
        )

    train_index = frame.index[dated & (values < cutoff)]
    eval_index = frame.index[dated & (values >= cutoff)]

    if len(train_index) == 0 or len(eval_index) == 0:
        earliest = values[dated].min()
        latest = values[dated].max()
        empty = "training" if len(train_index) == 0 else "evaluation"
        raise ValueError(
            f"cutoff {cutoff.date()} leaves the {empty} side empty for column "
            f"{date_column!r}, which spans {earliest.date()} to {latest.date()}. "
            f"Choose a cutoff inside that range."
        )

    return TemporalSplit(
        cutoff=cutoff,
        train_index=train_index,
        eval_index=eval_index,
        n_undated=n_undated,
        date_column=date_column,
        warnings=tuple(warnings),
    )


def walk_forward_folds(
    frame: pd.DataFrame,
    date_column: str,
    n_folds: int = 4,
    horizon: timedelta = timedelta(days=30),
) -> list[Fold]:
    """Expanding-window folds: each trains only on what preceded it.

    The training window grows and the evaluation window slides forward, which is
    how the model will actually be used — fitted on everything known so far,
    asked about what comes next. A ``KFold`` with ``shuffle=True`` is the same
    leak as a random split wearing cross-validation's clothes, and is banned by
    CLAUDE.md.
    """
    values = _dates(frame, date_column)
    dated = values.notna()
    if not dated.any():
        raise ValueError(f"column {date_column!r} holds no usable dates")

    latest = values[dated].max()
    earliest = values[dated].min()
    span = latest - earliest
    needed = horizon * n_folds

    if needed >= span:
        raise ValueError(
            f"{n_folds} folds of {horizon.days} day(s) need more than "
            f"{needed.days} days, but {date_column!r} spans only {span.days} "
            f"days ({earliest.date()} to {latest.date()}). Use fewer folds, a "
            f"shorter horizon, or more history."
        )

    folds: list[Fold] = []
    for i in range(n_folds):
        eval_start = latest - horizon * (n_folds - i)
        eval_end = eval_start + horizon
        train_index = frame.index[dated & (values < eval_start)]
        eval_index = frame.index[
            dated & (values >= eval_start) & (values < eval_end)
        ]
        if len(train_index) == 0:
            raise ValueError(
                f"fold {i} would train on nothing: no rows in {date_column!r} "
                f"fall before {eval_start.date()}. Use fewer folds or a shorter "
                f"horizon so the first fold has history to learn from."
            )
        folds.append(
            Fold(
                index=i,
                train_index=train_index,
                eval_index=eval_index,
                train_end=eval_start,
                eval_start=eval_start,
                eval_end=eval_end,
            )
        )
    return folds


def _sample(ids: Sequence[object]) -> tuple[str, ...]:
    return tuple(str(value) for value in list(ids)[:MAX_SAMPLE_IDS])


def label_with_censoring(
    frame: pd.DataFrame,
    *,
    id_column: str,
    start_column: str,
    event_column: str,
    observation_end: date | pd.Timestamp,
    horizon: timedelta,
    min_tenure: timedelta = timedelta(0),
    data_end: date | pd.Timestamp | None = None,
) -> LabelResult:
    """Label churn over a stated horizon, and say who could not be labelled.

    The observation window ends at ``observation_end``; the prediction horizon
    is the ``horizon`` that follows it. A customer is labelled 1 if their churn
    event falls inside that horizon, and 0 only if they were observed for the
    whole of it without one.

    Everyone else is dropped, with a reason:

    * churned on or before the window closed — that is history, not a
      prediction, and including them trains the model on outcomes it would
      already have known
    * still active but the data stops before the horizon does — **censored**
      (I3). Their outcome is unknown; calling it 0 would teach the model that
      the end of the dataset means the customer stayed
    * shorter than ``min_tenure`` at the window end — too new to have the
      history the features assume
    * started after the window closed — not in the population at all
    """
    window_end = pd.Timestamp(observation_end)
    horizon_end = window_end + horizon
    last_observed = window_end if data_end is None else pd.Timestamp(data_end)

    ids = frame[id_column]
    started = pd.to_datetime(frame[start_column])
    event = pd.to_datetime(frame[event_column])

    excluded: list[ExcludedRows] = []

    def drop(mask: pd.Series, reason: str) -> pd.Series:
        if mask.any():
            excluded.append(
                ExcludedRows(
                    reason=reason,
                    n=int(mask.sum()),
                    sample_ids=_sample(ids[mask]),
                )
            )
        return mask

    remaining = pd.Series(True, index=frame.index)

    after_window = drop(
        remaining & started.notna() & (started > window_end),
        f"started after the observation window closed on {window_end.date()}, "
        "so they were not a customer at the point a prediction would have been made",
    )
    remaining &= ~after_window

    tenure_short = drop(
        remaining & started.notna() & ((window_end - started) < min_tenure),
        f"less than {min_tenure.days} day(s) of history at "
        f"{window_end.date()}, which is below the minimum tenure the "
        "features assume",
    )
    remaining &= ~tenure_short

    undated_start = drop(
        remaining & started.isna(),
        f"no value in {start_column!r}, so tenure could not be established",
    )
    remaining &= ~undated_start

    already_gone = drop(
        remaining & event.notna() & (event <= window_end),
        f"churned on or before {window_end.date()}, inside the observation "
        "window rather than the prediction horizon — that is recorded history, "
        "not something to predict",
    )
    remaining &= ~already_gone

    churned = (
        remaining & event.notna() & (event > window_end) & (event <= horizon_end)
    )
    survived_event = remaining & event.notna() & (event > horizon_end)

    # No event on record. Only a 0 if the data actually watched them the whole
    # way through the horizon (I3).
    no_event = remaining & event.isna()
    observed_throughout = no_event & (last_observed >= horizon_end)
    censored_mask = no_event & (last_observed < horizon_end)

    labelled = churned | survived_event | observed_throughout
    labels = pd.Series(
        churned[labelled].astype(int).to_numpy(),
        index=pd.Index(ids[labelled], name=id_column),
        name="churned",
    )

    censored_index = pd.Index(ids[censored_mask], name=id_column)
    if len(censored_index):
        excluded.append(
            ExcludedRows(
                reason=(
                    f"censored — still active with no churn event, but the data "
                    f"ends {last_observed.date()}, before the horizon closes "
                    f"{horizon_end.date()}. Their outcome is unknown and is not "
                    f"a negative (I3)"
                ),
                n=len(censored_index),
                sample_ids=_sample(censored_index),
            )
        )

    return LabelResult(
        labels=labels,
        censored=censored_index,
        excluded=tuple(excluded),
        observation_end=window_end,
        horizon_end=horizon_end,
        n_input=len(frame),
    )


def assert_no_time_travel(
    source_dates: pd.DataFrame, prediction_dates: pd.Series
) -> None:
    """Raise if any feature's source data postdates the prediction it feeds (I2).

    ``source_dates`` holds one column per feature, each value the date that
    feature's underlying data was observed for that row. ``prediction_dates``
    is the date each row is a prediction *for*. A feature whose source date is
    later than its prediction date is using information that did not exist yet
    — the failure is silent, produces excellent validation metrics, and is
    invisible in the model artifact, which is why it gets an assertion rather
    than a warning.
    """
    predictions = pd.to_datetime(pd.Series(prediction_dates)).reset_index(drop=True)
    offenders: list[str] = []

    for column in source_dates.columns:
        observed = pd.to_datetime(source_dates[column]).reset_index(drop=True)
        late = observed.notna() & predictions.notna() & (observed > predictions)
        if not late.any():
            continue
        worst = int(observed.where(late).idxmax())
        offenders.append(
            f"{column!r}: {int(late.sum())} row(s) sourced after their "
            f"prediction date, worst at row {worst} where the data is dated "
            f"{observed[worst].date()} but the prediction is for "
            f"{predictions[worst].date()}"
        )

    if offenders:
        raise TimeTravelError(
            "features were computed from data that did not exist at prediction "
            "time (I2). " + "; ".join(offenders)
        )
