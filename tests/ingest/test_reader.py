"""The parser, checked against tests/fixtures/nasty/MANIFEST.md.

The manifest is the specification and it was written before this code existed.
Each test below names the fixture it covers and asserts the manifest's "correct
parse" clause; the "must not silently" clauses are asserted too, because a
parser that returns a clean-looking frame while destroying data passes any test
that only checks the frame is not empty.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from churnkit.ingest import read_table

NASTY = Path(__file__).resolve().parents[1] / "fixtures" / "nasty"

ALL_FIXTURES = sorted(p for p in NASTY.iterdir() if p.suffix in {".csv", ".tsv"})


def col(result, name):
    return result.columns[name]


# ── Rules that apply to all fifteen ───────────────────────────────────────────


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
def test_every_fixture_parses_without_raising(path):
    """A bad file is reported, never fatal. Fifteen files, zero exceptions."""
    result = read_table(path)
    assert isinstance(result.frame, pd.DataFrame)


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
def test_row_counts_reconcile(path):
    """MANIFEST: n_rows_in == n_rows_kept + n_rows_reported, for every file."""
    result = read_table(path)
    assert result.n_rows_in == result.n_rows_kept + result.n_rows_reported
    assert result.n_rows_kept == len(result.frame)


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
def test_every_column_carries_stats(path):
    """I10: a column with no stats cannot name itself when it goes wrong."""
    result = read_table(path)
    assert set(result.columns) == set(result.frame.columns)
    for name, stats in result.columns.items():
        assert stats.name == name
        assert stats.n_parsed + stats.n_failed + stats.n_missing == len(result.frame)


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.name)
def test_detection_is_always_reported(path):
    result = read_table(path)
    assert result.encoding
    assert result.delimiter
    assert result.header_row >= 0
    assert isinstance(result.warnings, list)


# ── 01 — delimiter is not a comma ─────────────────────────────────────────────


def test_01_semicolon_delimiter_and_commas_inside_values():
    result = read_table(NASTY / "01_semicolon.csv")

    assert result.delimiter == ";"
    assert list(result.frame.columns) == [
        "account_ref",
        "plan_name",
        "monthly_fee",
        "left_us",
    ]
    assert len(result.frame) == 8
    # The comma-frequency trap: values keep their internal commas.
    assert result.frame["plan_name"].iloc[1] == "Pro, annual, discounted"
    assert col(result, "monthly_fee").kind == "numeric"
    assert result.frame["monthly_fee"].iloc[0] == pytest.approx(19.99)


# ── 02 — tabs inside quoted fields ────────────────────────────────────────────


def test_02_quoted_tabs_are_data_not_separators():
    result = read_table(NASTY / "02_tab_quoted.tsv")

    assert result.delimiter == "\t"
    assert len(result.frame.columns) == 4
    assert len(result.frame) == 6
    assert result.frame["notes"].iloc[0] == "called support\ttwice\tin March"
    assert col(result, "spend_total").kind == "numeric"


# ── 03 — byte-order mark ──────────────────────────────────────────────────────


def test_03_bom_never_reaches_a_column_name():
    result = read_table(NASTY / "03_utf8_bom.csv")

    assert result.encoding == "utf-8-sig"
    assert list(result.frame.columns)[0] == "customer_key"
    assert "﻿" not in "".join(result.frame.columns)
    assert len(result.frame) == 6


# ── 04 — latin-1 ──────────────────────────────────────────────────────────────


def test_04_latin1_names_survive_intact():
    result = read_table(NASTY / "04_latin1.csv")

    assert result.encoding in {"latin-1", "cp1252"}
    assert list(result.frame["full_name"]) == [
        "José Álvarez",
        "Renée Dupré",
        "Jürgen Müller",
        "Sofía Núñez",
        "Anaïs Béranger",
    ]
    # No mojibake and no dropped rows — both corrupt customer identifiers.
    assert "�" not in "".join(result.frame["full_name"])
    assert len(result.frame) == 5


# ── 05 — cp1252 ───────────────────────────────────────────────────────────────


def test_05_cp1252_punctuation_is_not_decoded_as_latin1():
    result = read_table(NASTY / "05_cp1252_smart_quotes.csv")

    assert result.encoding == "cp1252"
    assert result.frame["last_note"].iloc[0] == "“price is too high”"
    assert result.frame["segment"].iloc[1] == "Mid–Market"
    # C1 control characters are what latin-1 would have produced here.
    assert not any(
        "\x80" <= ch <= "\x9f" for ch in "".join(result.frame["last_note"])
    )


# ── 06 — DD.MM.YYYY ───────────────────────────────────────────────────────────


def test_06_dotted_dates_are_day_first_for_the_whole_column():
    result = read_table(NASTY / "06_dates_dotted.csv")

    signup = result.frame["signup_date"]
    assert col(result, "signup_date").kind == "datetime"
    assert col(result, "signup_date").date_format == "day-first"
    assert signup.iloc[0].date() == date(2023, 3, 7)
    assert signup.iloc[2].date() == date(2023, 6, 30)
    # The rows where both components are <= 12 must follow the column, not a
    # per-row guess: 11.11.2022 is unambiguous, 01.09.2021 is the real test.
    assert signup.iloc[3].date() == date(2021, 9, 1)
    assert result.frame["last_seen_date"].iloc[0].date() == date(2023, 12, 25)


# ── 07 — Excel serials ────────────────────────────────────────────────────────


def test_07_excel_serials_convert_and_plain_integers_do_not():
    result = read_table(NASTY / "07_dates_excel_serial.csv")

    assert col(result, "start_date").kind == "datetime"
    assert col(result, "start_date").date_format == "excel-serial"
    assert result.frame["start_date"].iloc[0].date() == date(2023, 1, 1)
    assert result.frame["start_date"].iloc[1].date() == date(2022, 1, 1)
    assert result.frame["start_date"].iloc[3].date() == date(2020, 1, 1)
    assert result.frame["renewal_date"].iloc[0].date() == date(2024, 1, 1)

    # tenure_days = 365 must not become 1900-12-30.
    assert col(result, "claims_count").kind == "numeric"
    assert col(result, "tenure_days").kind == "numeric"
    assert result.frame["tenure_days"].iloc[0] == 365

    assert any("start_date" in w and "serial" in w for w in result.warnings)


# ── 08 — genuinely undecidable ────────────────────────────────────────────────


def test_08_ambiguous_dates_are_flagged_never_guessed():
    result = read_table(NASTY / "08_dates_ambiguous.csv")

    assert result.requires_disambiguation
    assert set(result.ambiguous_date_columns) == {"created_on", "closed_on"}
    assert col(result, "created_on").ambiguous_date_format
    # Raw values are preserved: guessing would be worse than not parsing.
    assert result.frame["created_on"].iloc[0] == "03/04/2024"
    assert col(result, "created_on").kind == "text"


def test_08_operator_can_resolve_the_ambiguity():
    """I7: inference proposes, a human decides — and the decision is honoured."""
    result = read_table(NASTY / "08_dates_ambiguous.csv", dayfirst=True)

    assert not result.requires_disambiguation
    assert col(result, "created_on").kind == "datetime"
    assert result.frame["created_on"].iloc[0].date() == date(2024, 4, 3)

    other = read_table(NASTY / "08_dates_ambiguous.csv", dayfirst=False)
    assert other.frame["created_on"].iloc[0].date() == date(2024, 3, 4)


# ── 09 — null spellings, one of which is data ─────────────────────────────────


def test_09_null_tokens_are_per_column_not_global():
    result = read_table(NASTY / "09_null_spellings.csv")

    assert len(result.frame) == 10

    # The trap: `None` is a real add-on tier in 4 of 10 rows.
    addon = col(result, "addon_tier")
    assert addon.kind == "text"
    assert addon.n_missing == 0
    assert (result.frame["addon_tier"] == "None").sum() == 4

    # ... while the numeric columns treat the same spellings as missing.
    calls = col(result, "support_calls")
    assert calls.kind == "numeric"
    assert calls.n_missing == 5
    assert calls.n_parsed == 5
    assert sorted(result.frame["support_calls"].dropna()) == [0.0, 2.0, 3.0, 4.0, 7.0]
    assert set(calls.null_tokens) >= {"NA", "null", "-", "?"}

    payment = col(result, "last_payment")
    assert payment.kind == "numeric"
    assert payment.n_missing == 5
    assert set(payment.null_tokens) >= {"N/A", "NULL", "--", "-"}

    # The operator is told the ambiguous column was left alone (I7).
    assert any("addon_tier" in w for w in result.warnings)


# ── 10 — numbers wearing costumes ─────────────────────────────────────────────


def test_10_currency_and_percent_are_scaled_and_recorded():
    result = read_table(NASTY / "10_currency_numerics.csv")

    order = col(result, "order_value")
    assert order.kind == "numeric"
    assert order.currency == "$"
    assert order.n_failed == 0
    assert list(result.frame["order_value"]) == pytest.approx(
        [1234.56, 98.00, 12000.00, 450.25, 7890.10, 64.99]
    )

    # 45% -> 0.45, not 45.0. A hundredfold error that reads as plausible.
    discount = col(result, "discount_rate")
    assert discount.unit == "percent"
    assert list(result.frame["discount_rate"]) == pytest.approx(
        [0.45, 0.05, 0.00, 0.125, 0.30, 1.00]
    )

    ltv = col(result, "lifetime_value")
    assert ltv.currency == "₹"
    assert list(result.frame["lifetime_value"]) == pytest.approx(
        [120000.0, 45000.0, 850000.0, 120000.0, 240000.0, 9999.0]
    )
    # Indian digit grouping and the lakh suffix are documented, not silent.
    assert any("lifetime_value" in w for w in result.warnings)


# ── 11 — the same header twice ────────────────────────────────────────────────


def test_11_duplicate_headers_both_survive():
    result = read_table(NASTY / "11_duplicate_columns.csv")

    assert list(result.frame.columns) == [
        "id",
        "amount",
        "amount_1",
        "region",
        "region_1",
        "status",
    ]
    # Older pandas kept the last occurrence and lost this value entirely.
    assert result.frame["amount"].iloc[0] == pytest.approx(100.00)
    assert result.frame["amount_1"].iloc[0] == pytest.approx(250.00)
    assert result.renamed_columns == {"amount_1": "amount", "region_1": "region"}
    assert any("amount_1" in w for w in result.warnings)
    assert len(result.frame) == 5


# ── 12 — junk above the header ────────────────────────────────────────────────


def test_12_real_header_found_below_the_export_banner():
    result = read_table(NASTY / "12_header_row_three.csv")

    assert result.header_row == 3
    assert list(result.frame.columns) == [
        "partner_code",
        "contract_value",
        "months_active",
        "renewed",
    ]
    assert len(result.frame) == 5
    assert col(result, "contract_value").kind == "numeric"
    # The preamble is reported, not discarded.
    assert any("CONFIDENTIAL" in line for line in result.preamble)
    assert any("header" in w.lower() for w in result.warnings)


# ── 13 — padding rows and a ghost column ──────────────────────────────────────


def test_13_ghost_column_and_empty_rows_are_dropped_out_loud():
    result = read_table(NASTY / "13_trailing_empty.csv")

    assert list(result.frame.columns) == ["site_id", "visits", "plan", "cancelled"]
    assert len(result.frame) == 4
    assert result.empty_rows_dropped == 3
    assert result.dropped_columns
    assert any("empty" in w.lower() for w in result.warnings)
    # The operator is not told the dataset is 7 rows.
    assert result.n_rows_kept == 4


# ── 14 — padding everywhere ───────────────────────────────────────────────────


def test_14_whitespace_never_creates_a_second_category():
    result = read_table(NASTY / "14_whitespace_padded.csv")

    assert list(result.frame.columns) == [
        "tenant_id",
        "plan_code",
        "months_billed",
        "is_gone",
    ]
    assert len(result.frame) == 5
    assert result.frame["tenant_id"].iloc[0] == "T1"
    # "  BASIC  " and "BASIC" are one category, not two.
    assert set(result.frame["plan_code"].dropna()) == {"BASIC", "PRO"}
    # Whitespace-only cells are missing, and counted.
    assert col(result, "plan_code").n_missing == 1
    assert col(result, "months_billed").kind == "numeric"
    assert col(result, "months_billed").n_missing == 1


# ── 15 — ragged rows ──────────────────────────────────────────────────────────


def test_15_ragged_rows_are_reported_line_by_line():
    result = read_table(NASTY / "15_ragged_rows.csv")

    assert result.n_rows_in == 8
    assert result.n_rows_kept == 4
    assert result.empty_rows_dropped == 1
    assert len(result.row_failures) == 3

    by_line = {f.line_number: f for f in result.row_failures}
    assert set(by_line) == {3, 4, 6}
    assert by_line[3].n_fields == 3
    assert by_line[4].n_fields == 5
    assert by_line[6].n_fields == 1
    # I10: the report shows the offending line, not just a count.
    assert "DV2" in by_line[3].raw_line
    assert "extra-unexpected-value" in by_line[4].raw_line

    # DV6's trailing empties are padding, so the row is recoverable.
    assert "DV6" in list(result.frame["device_id"])
    # A whole upload is not aborted over three bad lines.
    assert list(result.frame["device_id"]) == ["DV1", "DV4", "DV6", "DV7"]


# ── Failure reporting on values the parser cannot handle ──────────────────────


def test_unparseable_values_are_counted_with_samples_never_replaced(tmp_path):
    """I10 plus the S3 ban on silent coercion: name the column, show the value."""
    path = tmp_path / "mixed.csv"
    path.write_text(
        "ref,spend\n"
        "A,100.00\n"
        "B,200.00\n"
        "C,300.00\n"
        "D,400.00\n"
        "E,not-a-number\n"
        "F,500.00\n",
        encoding="utf-8",
    )

    result = read_table(path)
    spend = result.columns["spend"]

    assert spend.kind == "numeric"
    assert spend.n_failed == 1
    assert spend.n_parsed == 5
    assert "not-a-number" in spend.sample_failures
    assert any("spend" in w for w in result.warnings)
    # Reported, not dropped: the row is still there.
    assert len(result.frame) == 6


def test_a_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        read_table(tmp_path / "nope.csv")
    assert "nope.csv" in str(exc.value)


# ── Cases the corpus does not cover, found by probing the parser ──────────────


def test_european_decimal_comma_is_not_read_as_digit_grouping(tmp_path):
    """`1,5` is one and a half in most of Europe, and fifteen if you strip commas.

    The corpus has no decimal-comma fixture, so the first implementation passed
    every test while turning 1,5 into 15.0 — the same class of tenfold error the
    manifest bans for percent signs.
    """
    path = tmp_path / "euro.csv"
    path.write_text(
        "id;amount;fee\nA;1,5;1.234,56\nB;2,75;9.870,10\n", encoding="utf-8"
    )

    result = read_table(path)

    assert result.delimiter == ";"
    assert list(result.frame["amount"]) == pytest.approx([1.5, 2.75])
    assert list(result.frame["fee"]) == pytest.approx([1234.56, 9870.10])
    assert result.columns["amount"].decimal_separator == ","
    assert any("amount" in w and "decimal" in w for w in result.warnings)


def test_thousands_grouping_still_wins_where_that_is_the_evidence(tmp_path):
    path = tmp_path / "grouped.csv"
    path.write_text("id,amount\nA,\"1,234\"\nB,\"45,000\"\n", encoding="utf-8")

    result = read_table(path)

    assert list(result.frame["amount"]) == pytest.approx([1234.0, 45000.0])
    assert result.columns["amount"].decimal_separator == "."


def test_line_numbers_survive_a_quoted_newline(tmp_path):
    """A failure has to point at the line a person will find in their editor."""
    path = tmp_path / "wrapped.csv"
    path.write_text(
        'id,note,amt\nA,"line one\nline two",5\nB,plain,6\nC,short\n',
        encoding="utf-8",
    )

    result = read_table(path)

    assert len(result.row_failures) == 1
    assert result.row_failures[0].line_number == 5
    assert result.row_failures[0].raw_line == "C,short"


def test_a_header_with_no_rows_says_so(tmp_path):
    path = tmp_path / "empty_table.csv"
    path.write_text("id,amount\n", encoding="utf-8")

    result = read_table(path)

    assert result.n_rows_kept == 0
    assert any("no data rows" in w for w in result.warnings)


def test_a_wrapped_row_is_reported_at_the_line_it_starts_on(tmp_path):
    """csv.reader counts the line a record ends on; people count where it starts."""
    path = tmp_path / "wrapped_failure.csv"
    path.write_text(
        'id,note,amt\nA,"line one\nline two"\nB,plain,6\n',
        encoding="utf-8",
    )

    result = read_table(path)

    assert [f.line_number for f in result.row_failures] == [2]


def test_space_grouped_numbers_parse_whatever_kind_of_space_is_used(tmp_path):
    """French exports group with spaces — ordinary, non-breaking, or narrow."""
    path = tmp_path / "spaced.csv"
    path.write_text(
        "id;montant\nA;1 234,56\nB;9 870,10\nC;2 500,00\n",
        encoding="utf-8",
    )

    result = read_table(path)

    assert list(result.frame["montant"]) == pytest.approx([1234.56, 9870.10, 2500.00])
    assert result.columns["montant"].n_failed == 0
