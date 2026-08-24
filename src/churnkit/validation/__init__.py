"""Validation that respects the arrow of time.

Platform code: no column name, category or business rule from any particular
dataset may appear here (I11).
"""

from churnkit.validation.temporal import (
    ExcludedRows,
    Fold,
    LabelResult,
    TemporalSplit,
    TimeTravelError,
    assert_no_time_travel,
    cutoff_split,
    label_with_censoring,
    walk_forward_folds,
)

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
