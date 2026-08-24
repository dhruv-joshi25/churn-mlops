"""Schema inference and leakage detection, checked against
tests/fixtures/leaky/MANIFEST.md.

The manifest is the specification and it was written before this code existed.
Each test names the fixture it covers and asserts both halves of that fixture's
entry: what must be caught, and what must **not** fire. The second half is the
one that matters. A detector that flags every column catches every leak and is
worthless, because an operator who sees warnings on clean data learns to click
through the one that was real.
"""

from datetime import date
from pathlib import Path

import pytest

from churnkit.ingest import read_table
from churnkit.ingest.infer import (
    LeakageOverrideRequired,
    UnconfirmedProposalError,
    infer_schema,
)

LEAKY = Path(__file__).resolve().parents[1] / "fixtures" / "leaky"

TARGETS = {
    "01_cancellation_date.csv": "did_leave",
    "02_perfect_predictor.csv": "lapsed",
    "03_null_mask_leak.csv": "attrited",
    "04_post_window_dates.csv": "not_renewed",
    "05_constant_within_class.csv": "stopped",
    "06_clean_baseline.csv": "left_service",
}


def propose(name, **kwargs):
    return infer_schema(read_table(LEAKY / name), **kwargs)


def flagged(proposal, column):
    return [f for f in proposal.leakage if f.column == column]


def rules_on(proposal, column):
    return {f.rule for f in flagged(proposal, column)}


# ── I7 — inference proposes, humans decide ────────────────────────────────────


def test_a_proposal_cannot_reach_training_on_its_own():
    """I7: no code path from inferred schema to training without confirmation."""
    proposal = propose("06_clean_baseline.csv")
    with pytest.raises(UnconfirmedProposalError):
        proposal.to_mapping()


def test_confirmation_produces_a_mapping():
    proposal = propose("06_clean_baseline.csv")
    mapping = proposal.confirm(
        target="left_service",
        id_column="customer_key",
        timestamp_column="signed_up_on",
    )
    assert mapping.target == "left_service"
    assert mapping.id_column == "customer_key"


def test_a_blocking_finding_cannot_be_confirmed_without_the_typed_override():
    """I5: detection halts the run; override is typed and logged, never a flag."""
    proposal = propose("01_cancellation_date.csv")
    assert proposal.requires_typed_override
    with pytest.raises(LeakageOverrideRequired):
        proposal.confirm(
            target="did_leave", id_column="member_ref", timestamp_column=None
        )


def test_the_typed_override_is_recorded_with_what_it_overrode():
    proposal = propose("01_cancellation_date.csv")
    mapping = proposal.confirm(
        target="did_leave",
        id_column="member_ref",
        timestamp_column=None,
        override_phrase=proposal.override_phrase,
    )
    assert mapping.overridden_findings
    assert "cancellation_date" in {f.column for f in mapping.overridden_findings}


def test_a_wrong_override_phrase_is_refused():
    proposal = propose("01_cancellation_date.csv")
    with pytest.raises(LeakageOverrideRequired):
        proposal.confirm(
            target="did_leave",
            id_column="member_ref",
            timestamp_column=None,
            override_phrase="yes",
        )


# ── 01 — the planted cancellation_date leak ───────────────────────────────────


def test_01_cancellation_date_is_caught_by_two_independent_rules():
    p = propose("01_cancellation_date.csv", target="did_leave")
    assert {"name_pattern", "null_mask"} <= rules_on(p, "cancellation_date")
    assert any(f.severity == "blocking" for f in flagged(p, "cancellation_date"))


def test_01_ordinary_features_are_left_alone():
    p = propose("01_cancellation_date.csv", target="did_leave")
    assert not flagged(p, "months_active")
    assert not flagged(p, "monthly_spend")


# ── 02 — perfect separator, innocent name ─────────────────────────────────────


def test_02_perfect_separation_blocks_on_statistics_alone():
    p = propose("02_perfect_predictor.csv", target="lapsed")
    rules = rules_on(p, "engagement_index")
    assert "single_column_auc" in rules
    assert "name_pattern" not in rules, "the name list is too wide"
    assert any(f.severity == "blocking" for f in flagged(p, "engagement_index"))


def test_02_the_evidence_quotes_the_auc():
    p = propose("02_perfect_predictor.csv", target="lapsed")
    finding = next(
        f for f in flagged(p, "engagement_index") if f.rule == "single_column_auc"
    )
    assert "1.00" in finding.evidence


# ── 03 — the missingness is the leak ──────────────────────────────────────────


def test_03_null_mask_leak_is_caught_though_the_values_are_noise():
    p = propose("03_null_mask_leak.csv", target="attrited")
    assert "null_mask" in rules_on(p, "followup_notes_len")
    assert any(f.severity == "blocking" for f in flagged(p, "followup_notes_len"))


# ── 04 — knowing the future ───────────────────────────────────────────────────


def test_04_dates_after_the_window_are_caught_when_the_window_is_known():
    p = propose(
        "04_post_window_dates.csv",
        target="not_renewed",
        observation_end=date(2024, 6, 30),
    )
    assert "post_window_datetime" in rules_on(p, "last_seen_on")
    assert not flagged(p, "signup_on"), "signup_on is inside the window"


def test_04_without_a_window_the_check_is_skipped_not_passed():
    """An unrun check reported as a pass is the silent-success failure mode."""
    p = propose("04_post_window_dates.csv", target="not_renewed")
    assert not flagged(p, "last_seen_on")
    skipped = {s.rule for s in p.skipped_checks}
    assert "post_window_datetime" in skipped


def test_04_trips_no_statistical_rule():
    p = propose("04_post_window_dates.csv", target="not_renewed")
    statistical = {"single_column_auc", "null_mask", "constant_within_class"}
    assert not any(f.rule in statistical for f in p.leakage)


# ── 05 — separation without a number ──────────────────────────────────────────


def test_05_constant_within_class_is_caught():
    p = propose("05_constant_within_class.csv", target="stopped")
    assert "constant_within_class" in rules_on(p, "retention_code")
    assert any(f.severity == "blocking" for f in flagged(p, "retention_code"))


def test_05_columns_that_vary_within_both_classes_are_left_alone():
    p = propose("05_constant_within_class.csv", target="stopped")
    assert not flagged(p, "billing_mode")
    assert not flagged(p, "city")


# ── 06 — the false-positive control ───────────────────────────────────────────


def test_06_the_clean_file_produces_no_blocking_findings():
    """The most important test in the file. See the manifest entry for why."""
    p = propose("06_clean_baseline.csv", target="left_service")
    blocking = [f for f in p.leakage if f.severity == "blocking"]
    assert blocking == [], f"false positives on clean data: {blocking}"


def test_06_the_clean_file_can_be_confirmed_without_an_override():
    p = propose("06_clean_baseline.csv", target="left_service")
    assert not p.requires_typed_override
    p.confirm(
        target="left_service",
        id_column="customer_key",
        timestamp_column="signed_up_on",
    )


# ── Thresholds — the tiering, asserted directly ───────────────────────────────


def test_a_globally_constant_column_is_not_a_constant_within_class_leak():
    """tenant_label has one value throughout. Useless, but not a leak."""
    p = propose("07_roles.csv", target="is_gone")
    assert "constant_within_class" not in rules_on(p, "tenant_label")


def test_weak_name_patterns_only_warn(tmp_path):
    """marital_status is a demographic feature, not a leak."""
    path = tmp_path / "demo.csv"
    # marital_status must vary independently of the target, or the fixture
    # plants a real leak and stops testing what it claims to test.
    rows = [
        f"D{i},{['single', 'married'][i % 2]},{20 + i % 40},"
        f"{1 if (i * 7) % 11 < 4 else 0}"
        for i in range(1, 61)
    ]
    path.write_text(
        "ref,marital_status,age,churned\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    p = infer_schema(read_table(path), target="churned")
    findings = flagged(p, "marital_status")
    assert all(f.severity == "warning" for f in findings)
    assert not p.requires_typed_override


def test_a_small_sample_downgrades_a_borderline_auc_to_a_warning(tmp_path):
    """At n=12 an AUC of 0.96 is luck, not evidence."""
    path = tmp_path / "tiny.csv"
    # One inversion in twelve rows: AUC 0.972, inside the 0.95-0.99 band and
    # short of the perfect separation that blocks regardless of sample size.
    slope = [3, 6, 9, 12, 15, 20, 18, 24, 27, 30, 33, 36]
    rows = [f"T{i + 1},{slope[i]},{1 if i >= 6 else 0}" for i in range(12)]
    path.write_text("ref,slope,gone\n" + "\n".join(rows) + "\n", encoding="utf-8")
    p = infer_schema(read_table(path), target="gone")
    assert not any(f.severity == "blocking" for f in flagged(p, "slope"))


# ── Part A — role inference ───────────────────────────────────────────────────

EXPECTED_ROLES = {
    "row_uuid": "identifier",
    "country_code": "categorical_low",
    "free_note": "free_text",
    "opened_at": "datetime",
    "score": "numeric",
    "tenant_label": "constant",
    "sku_reference": "categorical_high",
    "is_gone": "target_candidate",
}


@pytest.mark.parametrize("column,expected", sorted(EXPECTED_ROLES.items()))
def test_07_each_column_gets_its_role(column, expected):
    p = propose("07_roles.csv")
    assert p.roles[column].role == expected


def test_07_every_role_carries_a_confidence_and_a_reason():
    p = propose("07_roles.csv")
    for role in p.roles.values():
        assert 0.0 <= role.confidence <= 1.0
        assert role.reasoning.strip()


def test_07_only_one_column_is_proposed_as_the_identifier():
    """sku_reference is near-unique too; the proposal must still pick one."""
    p = propose("07_roles.csv")
    identifiers = [n for n, r in p.roles.items() if r.role == "identifier"]
    assert identifiers == ["row_uuid"]


# ── Target, id and timestamp proposals ────────────────────────────────────────


@pytest.mark.parametrize("name,target", sorted(TARGETS.items()))
def test_the_target_is_ranked_first_across_the_corpus(name, target):
    p = propose(name)
    assert p.target_candidates
    assert p.target_candidates[0].name == target


def test_the_timestamp_column_is_proposed_where_one_exists():
    p = propose("06_clean_baseline.csv")
    assert p.timestamp_column == "signed_up_on"


def test_no_timestamp_is_reported_honestly_rather_than_invented():
    """I1: a dataset with no time column is marked, not quietly given one."""
    p = propose("05_constant_within_class.csv")
    assert p.timestamp_column is None
    assert any("no timestamp" in w.lower() for w in p.warnings)


def test_every_exclusion_carries_a_reason():
    p = propose("07_roles.csv")
    for column, reason in p.excluded.items():
        assert reason.strip(), f"{column} excluded without a reason"


# ── 08 — a dataset with no English in it at all ───────────────────────────────
#
# The platform promise is that any company points this at their own data. A
# German gym's export has no English column names, "ja"/"nein" instead of
# yes/no, and "nr" instead of id. Nothing here may depend on the operator
# happening to speak English (I11 in spirit: no dataset's conventions baked in).


def test_08_a_non_english_target_is_still_proposed():
    p = propose("08_non_english.csv")
    assert p.target == "gekündigt"


def test_08_a_non_english_identifier_is_still_proposed():
    p = propose("08_non_english.csv")
    assert p.id_column == "mitglied_nr"
    assert p.roles["mitglied_nr"].role == "identifier"


def test_08_a_guess_with_no_name_evidence_says_it_is_a_guess():
    """Confidence has to fall when the only evidence is "it has two values"."""
    p = propose("08_non_english.csv")
    top = p.target_candidates[0]
    assert top.name == "gekündigt"
    assert top.confidence < 0.6
    assert any("confidence" in w.lower() or "guess" in w.lower() for w in p.warnings)


def test_08_the_rest_of_the_columns_still_get_sensible_roles():
    p = propose("08_non_english.csv")
    assert p.roles["standort"].role == "categorical_low"
    assert p.roles["beitrag_eur"].role == "numeric"
    assert p.roles["vertrag_beginn"].role == "datetime"


def test_a_binary_column_that_is_not_the_target_keeps_its_own_role():
    """Telco has seven binary columns; only one of them is the churn target."""
    p = propose("06_clean_baseline.csv", target="left_service")
    assert p.roles["district"].role == "categorical_low"
