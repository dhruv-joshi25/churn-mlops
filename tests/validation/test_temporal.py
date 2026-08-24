"""Temporal validation — the module the whole project rests on.

CLAUDE.md bans random splits on data with a time dimension, and the reason is
not stylistic: a random split trains on the future and evaluates on the past,
which flatters every metric it touches. The test that matters most in this file
is `test_a_random_split_flatters_the_model`, which builds a dataset where the
churn driver changes over time and asserts that the temporal number — the
pessimistic, correct one — is what the module reports.

Nothing here imports train_test_split. The random split it compares against is
built by hand, precisely so that the structural guard in test_layout_guards.py
can ban the import outright.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from churnkit.validation.temporal import (
    TimeTravelError,
    assert_no_time_travel,
    cutoff_split,
    label_with_censoring,
    walk_forward_folds,
)

SEED = 20240824


def drifting_dataset(n=1200):
    """Churn driven by one feature early and a different one later.

    A model that trains on the late period predicts the late period well. A
    random split hands it exactly that; a cutoff split does not. The gap between
    the two is the thing this project exists to stop reporting.
    """
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp("2023-01-01")
    days = rng.integers(0, 720, size=n)
    observed = start + pd.to_timedelta(days, unit="D")
    early = days < 360

    feature_a = rng.normal(size=n)
    feature_b = rng.normal(size=n)
    # Early churn follows A, late churn follows B. Same columns, different world.
    logit = np.where(early, 2.4 * feature_a, 2.4 * feature_b)
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame(
        {
            "customer": [f"C{i:05d}" for i in range(n)],
            "observed_on": observed,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "churned": churned.astype(int),
        }
    ).sort_values("observed_on", ignore_index=True)


def fit_predict_auc(train, evaluate):
    """A deliberately plain model. The point is the split, not the estimator."""
    from sklearn.linear_model import LogisticRegression

    cols = ["feature_a", "feature_b"]
    model = LogisticRegression().fit(train[cols], train["churned"])
    scores = model.predict_proba(evaluate[cols])[:, 1]
    y = evaluate["churned"].to_numpy()
    if len(set(y)) < 2:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy()
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ── The headline claim ────────────────────────────────────────────────────────


def test_a_random_split_flatters_the_model():
    """The whole argument for this module, as an executable assertion."""
    df = drifting_dataset()
    cutoff = pd.Timestamp("2024-01-01")

    split = cutoff_split(df, "observed_on", cutoff)
    temporal_auc = fit_predict_auc(df.loc[split.train_index], df.loc[split.eval_index])

    # A random split of the same size, built by hand — see the module docstring.
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(len(df))
    n_train = len(split.train_index)
    random_auc = fit_predict_auc(
        df.iloc[shuffled[:n_train]], df.iloc[shuffled[n_train:]]
    )

    assert random_auc - temporal_auc > 0.05, (
        f"the fixture must actually drift: random {random_auc:.3f} vs temporal "
        f"{temporal_auc:.3f}"
    )
    assert split.reported_auc_is_temporal


# ── cutoff_split ──────────────────────────────────────────────────────────────


def test_the_cutoff_is_strict_on_the_training_side():
    """A row dated exactly on the cutoff belongs to evaluation, never training."""
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "y": [0, 1, 0],
        }
    )
    split = cutoff_split(df, "when", pd.Timestamp("2024-02-01"))
    assert list(split.train_index) == [0]
    assert list(split.eval_index) == [1, 2]


def test_an_empty_side_names_the_column_and_the_dates():
    df = pd.DataFrame({"when": pd.to_datetime(["2024-05-01", "2024-06-01"])})
    with pytest.raises(ValueError) as exc:
        cutoff_split(df, "when", pd.Timestamp("2024-01-01"))
    message = str(exc.value)
    assert "when" in message
    assert "2024-01-01" in message
    assert "2024-05-01" in message


def test_rows_with_no_date_are_excluded_and_counted_not_silently_kept():
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2024-01-01", None, "2024-03-01"]),
            "y": [0, 1, 0],
        }
    )
    split = cutoff_split(df, "when", pd.Timestamp("2024-02-01"))
    assert 1 not in split.train_index and 1 not in split.eval_index
    assert split.n_undated == 1
    assert any("no date" in w.lower() for w in split.warnings)


# ── walk_forward_folds ────────────────────────────────────────────────────────


def test_folds_never_train_on_their_own_evaluation_window():
    df = drifting_dataset(600)
    folds = walk_forward_folds(df, "observed_on", n_folds=4, horizon=timedelta(days=60))
    assert len(folds) == 4
    for fold in folds:
        train_dates = df.loc[fold.train_index, "observed_on"]
        assert train_dates.max() < fold.eval_start


def test_the_training_window_expands(): 
    df = drifting_dataset(600)
    folds = walk_forward_folds(df, "observed_on", n_folds=4, horizon=timedelta(days=60))
    sizes = [len(f.train_index) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_folds_do_not_overlap_each_other():
    df = drifting_dataset(600)
    folds = walk_forward_folds(df, "observed_on", n_folds=3, horizon=timedelta(days=60))
    seen: set[int] = set()
    for fold in folds:
        current = set(fold.eval_index)
        assert not (current & seen)
        seen |= current


def test_too_few_folds_for_the_data_is_refused_with_the_numbers():
    df = drifting_dataset(50)
    # 40 folds x 60 days needs 2400 days of history; the fixture spans ~720.
    with pytest.raises(ValueError) as exc:
        walk_forward_folds(df, "observed_on", n_folds=40, horizon=timedelta(days=60))
    message = str(exc.value)
    assert "observed_on" in message
    assert "2400" in message


# ── label_with_censoring (I3) ─────────────────────────────────────────────────


def labelling_frame():
    return pd.DataFrame(
        {
            "cust": ["A", "B", "C", "D", "E", "F"],
            "joined": pd.to_datetime(
                [
                    "2023-01-01",  # A churns inside the horizon
                    "2023-01-01",  # B still active, observed throughout
                    "2024-05-15",  # C too new at the window end
                    "2023-01-01",  # D churned before the window closed
                    "2023-01-01",  # E active but data stops early -> censored
                    "2024-08-01",  # F joined after the window
                ]
            ),
            "left_on": pd.to_datetime(
                [None, None, None, "2024-03-01", None, None]
            ).where(
                pd.Series([False, False, False, True, False, False]), pd.NaT
            ),
        }
    )


def build_labels(data_end="2024-09-30"):
    frame = labelling_frame()
    frame.loc[3, "left_on"] = pd.Timestamp("2024-03-01")
    frame.loc[0, "left_on"] = pd.Timestamp("2024-08-15")
    return label_with_censoring(
        frame,
        id_column="cust",
        start_column="joined",
        event_column="left_on",
        observation_end=date(2024, 6, 30),
        horizon=timedelta(days=90),
        min_tenure=timedelta(days=90),
        data_end=pd.Timestamp(data_end),
    )


def test_a_customer_who_churned_inside_the_horizon_is_labelled_one():
    assert build_labels().labels["A"] == 1


def test_a_customer_observed_through_the_horizon_is_labelled_zero():
    assert build_labels().labels["B"] == 0


def test_an_active_customer_who_was_not_observed_long_enough_is_censored():
    """I3: censored is not the same as negative, and must never be labelled 0."""
    result = build_labels(data_end="2024-08-01")
    assert "E" not in result.labels
    assert "E" in set(result.censored)


def test_a_customer_below_the_minimum_tenure_is_excluded_with_a_reason():
    result = build_labels()
    assert "C" not in result.labels
    reasons = {e.reason for e in result.excluded}
    assert any("tenure" in r.lower() for r in reasons)


def test_a_customer_who_churned_before_the_window_closed_is_excluded():
    result = build_labels()
    assert "D" not in result.labels


def test_every_excluded_row_is_counted_and_sampled():
    result = build_labels()
    for excluded in result.excluded:
        assert excluded.n > 0
        assert excluded.sample_ids
        assert excluded.reason.strip()


def test_the_counts_reconcile():
    result = build_labels()
    dropped = sum(e.n for e in result.excluded)
    assert result.n_input == len(result.labels) + len(result.censored) + dropped


def test_censoring_is_never_silently_folded_into_the_negative_class():
    a = build_labels(data_end="2024-09-30").labels
    b = build_labels(data_end="2024-08-01").labels
    assert (b == 0).sum() < (a == 0).sum()


# ── assert_no_time_travel (I2) ────────────────────────────────────────────────


def test_a_feature_sourced_after_its_prediction_date_raises():
    source = pd.DataFrame(
        {
            "spend_30d": pd.to_datetime(["2024-01-01", "2024-05-01"]),
            "logins_30d": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        }
    )
    predict_on = pd.Series(pd.to_datetime(["2024-02-01", "2024-02-01"]))
    with pytest.raises(TimeTravelError) as exc:
        assert_no_time_travel(source, predict_on)
    message = str(exc.value)
    assert "spend_30d" in message
    assert "2024-05-01" in message
    assert "logins_30d" not in message


def test_data_exactly_on_the_prediction_date_is_allowed():
    source = pd.DataFrame({"f": pd.to_datetime(["2024-02-01"])})
    assert_no_time_travel(source, pd.Series(pd.to_datetime(["2024-02-01"])))


def test_a_clean_feature_matrix_passes_quietly():
    source = pd.DataFrame({"f": pd.to_datetime(["2024-01-01", "2024-01-15"])})
    assert_no_time_travel(source, pd.Series(pd.to_datetime(["2024-02-01"] * 2)))
