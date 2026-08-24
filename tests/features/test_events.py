"""Point-in-time features from an event log.

I2 is the whole point of this module. A snapshot table says a customer makes 15
logins a month; an event log says they fell from 15 to 3, and that fall is the
signal. Computing it means reaching backwards through history, and reaching one
day too far forwards produces a feature that predicts the future because it
already contains it.

The tests that matter most here are the ones with an event planted on and after
the prediction date. A naive `groupby(customer).agg(...)` swallows both and
produces a beautiful, worthless model.
"""

from datetime import timedelta

import pandas as pd
import pytest

from churnkit.features import events as events_module
from churnkit.features.events import FEATURE_GROUPS, build_features
from churnkit.validation.temporal import TimeTravelError

AS_OF = pd.Timestamp("2024-06-01")


def log(rows):
    return pd.DataFrame(
        rows, columns=["customer", "occurred_at", "event_type", "value"]
    ).assign(occurred_at=lambda d: pd.to_datetime(d["occurred_at"]))


def build(frame, as_of=AS_OF, **kwargs):
    return build_features(
        frame,
        customer_column="customer",
        timestamp_column="occurred_at",
        prediction_dates=as_of,
        type_column="event_type",
        value_column="value",
        **kwargs,
    )


# ── I2 — the reason this module has tests at all ──────────────────────────────


def test_an_event_on_the_prediction_date_is_not_used():
    """Strictly before. A same-day event is not knowable when scoring that day."""
    frame = log(
        [
            ("A", "2024-05-30", "login", 1.0),
            ("A", "2024-06-01", "login", 1.0),  # exactly on the date
        ]
    )
    out = build(frame)
    assert out.features.loc[("A", AS_OF), "events_7d"] == 1


def test_an_event_after_the_prediction_date_is_not_used():
    frame = log(
        [
            ("A", "2024-05-30", "login", 1.0),
            ("A", "2024-07-15", "login", 1.0),  # the future
        ]
    )
    out = build(frame)
    assert out.features.loc[("A", AS_OF), "events_30d"] == 1


def test_a_naive_aggregation_would_have_counted_more():
    """The failure this module exists to prevent, stated as a comparison."""
    frame = log(
        [
            ("A", "2024-05-30", "login", 1.0),
            ("A", "2024-06-01", "login", 1.0),
            ("A", "2024-07-15", "login", 1.0),
        ]
    )
    naive = len(frame)
    ours = build(frame).features.loc[("A", AS_OF), "events_90d"]
    assert ours == 1
    assert naive > ours


def test_the_time_travel_assertion_runs_inside_the_build(monkeypatch):
    """S6: call it in the generation path, not only in tests."""
    called = {}

    def spy(source_dates, prediction_dates):
        called["yes"] = True

    monkeypatch.setattr(events_module, "assert_no_time_travel", spy)
    build(log([("A", "2024-05-30", "login", 1.0)]))
    assert called.get("yes")


def test_a_source_date_that_postdates_its_prediction_is_fatal():
    """If the windowing is ever broken, the build must fail rather than ship."""
    frame = log([("A", "2024-05-30", "login", 1.0)])
    out = build(frame)
    broken = out.source_dates.copy()
    broken.iloc[0, 0] = pd.Timestamp("2025-01-01")
    with pytest.raises(TimeTravelError):
        events_module.assert_no_time_travel(
            broken, pd.Series([AS_OF] * len(broken))
        )


# ── Windows, deltas, recency ──────────────────────────────────────────────────


def declining():
    """15 events in the older month, 3 in the recent one — a customer leaving."""
    rows = []
    for i in range(15):
        rows.append(("A", AS_OF - timedelta(days=35 + i), "login", 2.0))
    for i in range(3):
        rows.append(("A", AS_OF - timedelta(days=3 + i * 5), "login", 2.0))
    return log(rows)


def test_rolling_counts_respect_their_window():
    out = build(declining()).features.loc[("A", AS_OF)]
    assert out["events_7d"] == 1
    assert out["events_30d"] == 3
    assert out["events_90d"] == 18


def test_window_over_window_delta_is_negative_for_a_declining_customer():
    out = build(declining()).features.loc[("A", AS_OF)]
    assert out["events_30d_delta"] < 0
    assert out["events_30d_ratio"] < 1.0


def test_recency_counts_days_since_the_last_event():
    out = build(declining()).features.loc[("A", AS_OF)]
    assert out["days_since_last_event"] == 3


def test_recency_is_reported_per_event_type():
    frame = log(
        [
            ("A", "2024-05-30", "login", 1.0),
            ("A", "2024-04-01", "purchase", 20.0),
        ]
    )
    out = build(frame).features.loc[("A", AS_OF)]
    assert out["days_since_last_login"] == 2
    assert out["days_since_last_purchase"] == 61


def test_the_usage_drop_flag_fires_on_a_thirty_percent_fall():
    out = build(declining()).features.loc[("A", AS_OF)]
    assert out["usage_down_30pct_mom"] == 1


def test_the_usage_drop_flag_stays_off_for_a_steady_customer():
    rows = [("A", AS_OF - timedelta(days=i * 3), "login", 1.0) for i in range(1, 21)]
    out = build(log(rows)).features.loc[("A", AS_OF)]
    assert out["usage_down_30pct_mom"] == 0


# ── Monetary, frequency, trend, gaps ──────────────────────────────────────────


def test_monetary_sums_and_means_per_window():
    frame = log(
        [
            ("A", "2024-05-30", "purchase", 10.0),
            ("A", "2024-05-29", "purchase", 30.0),
        ]
    )
    out = build(frame).features.loc[("A", AS_OF)]
    assert out["value_sum_7d"] == 40.0
    assert out["value_mean_7d"] == 20.0


def test_frequency_is_events_per_active_day():
    frame = log(
        [
            ("A", "2024-05-30", "login", 1.0),
            ("A", "2024-05-30", "login", 1.0),
            ("A", "2024-05-29", "login", 1.0),
        ]
    )
    out = build(frame).features.loc[("A", AS_OF)]
    assert out["events_per_active_day"] == pytest.approx(1.5)


def test_the_engagement_slope_is_negative_when_activity_decays():
    out = build(declining()).features.loc[("A", AS_OF)]
    assert out["engagement_slope_90d"] < 0


def test_the_longest_inactivity_gap_is_found():
    frame = log(
        [
            ("A", AS_OF - timedelta(days=80), "login", 1.0),
            ("A", AS_OF - timedelta(days=20), "login", 1.0),
            ("A", AS_OF - timedelta(days=18), "login", 1.0),
        ]
    )
    out = build(frame).features.loc[("A", AS_OF)]
    assert out["longest_gap_90d"] == 60


# ── Multiple customers and dates ──────────────────────────────────────────────


def test_each_customer_gets_their_own_row():
    frame = log(
        [
            ("A", "2024-05-30", "login", 1.0),
            ("B", "2024-05-20", "login", 1.0),
            ("B", "2024-05-21", "login", 1.0),
        ]
    )
    out = build(frame)
    assert out.features.loc[("A", AS_OF), "events_30d"] == 1
    assert out.features.loc[("B", AS_OF), "events_30d"] == 2


def test_per_customer_prediction_dates_are_honoured():
    frame = log(
        [
            ("A", "2024-03-15", "login", 1.0),
            ("B", "2024-03-15", "login", 1.0),
        ]
    )
    dates = pd.Series(
        {"A": pd.Timestamp("2024-03-20"), "B": pd.Timestamp("2024-06-01")}
    )
    out = build_features(
        frame,
        customer_column="customer",
        timestamp_column="occurred_at",
        prediction_dates=dates,
        type_column="event_type",
        value_column="value",
    )
    assert out.features.loc[("A", pd.Timestamp("2024-03-20")), "events_7d"] == 1
    assert out.features.loc[("B", pd.Timestamp("2024-06-01")), "events_7d"] == 0


# ── The benchmarking hook S6 asks for ─────────────────────────────────────────


def test_feature_groups_can_be_selected_for_benchmarking():
    frame = declining()
    everything = build(frame).features
    counts_only = build(frame, groups=("counts",)).features
    assert len(counts_only.columns) < len(everything.columns)
    assert "events_30d" in counts_only.columns
    assert "engagement_slope_90d" not in counts_only.columns


def test_the_group_list_is_discoverable():
    assert "counts" in FEATURE_GROUPS
    assert "trend" in FEATURE_GROUPS


def test_an_unknown_group_is_refused_by_name():
    with pytest.raises(ValueError) as exc:
        build(declining(), groups=("nonsense",))
    assert "nonsense" in str(exc.value)
