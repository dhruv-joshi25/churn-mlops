"""Read an operator's file without lying about what was in it.

The contract this module keeps is narrow and absolute: **a value that cannot be
parsed is counted and reported, never quietly replaced or dropped**. Every
detection it performs — encoding, delimiter, header row, date order, null
spellings, currency notation — is returned in the :class:`ParseResult` rather
than applied invisibly, because each one is a guess that can be wrong, and a
wrong guess the operator cannot see is indistinguishable from correct parsing
until a model has been trained on it.

``tests/fixtures/nasty/MANIFEST.md`` is the specification. It was written before
this module and lists, per failure mode, what a correct parse produces and what
the parser must never do silently.

Two deliberate non-behaviours, both from that manifest:

* Ambiguous dates are **not** guessed. ``03/04/2024`` with no counter-evidence
  in the column returns flagged and unparsed; a locale default here silently
  reorders a whole column (I7 — inference proposes, a human decides).
* Null spellings are applied **per column**, never globally. ``None`` is a real
  category in a text column and missing data in a numeric one, and a global
  token list deletes the former to serve the latter.

No column name, category or business rule from any particular dataset appears
here; everything is inferred from the file in front of it (I11).
"""

from __future__ import annotations

import codecs
import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

__all__ = ["ColumnStats", "ParseResult", "RowFailure", "read_table"]

# ── Detection tables ──────────────────────────────────────────────────────────

CANDIDATE_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")

# Spellings of "missing" seen in exported CSVs. Matched case-insensitively, and
# only ever inside a column that is otherwise numeric or a date — see
# _null_tokens_in for why that qualification is the whole point.
NULL_TOKENS: frozenset[str] = frozenset(
    {"na", "n/a", "null", "none", "nan", "nil", "-", "--", "?", "unknown"}
)

CURRENCY_SYMBOLS: str = "$€£¥₹₩₽"

# Indian numbering suffixes. Applied only where a currency symbol is present, so
# a plain "1.2L" in a product-code column is not multiplied by a hundred
# thousand on the strength of one letter.
MULTIPLIER_SUFFIXES: dict[str, float] = {
    "l": 1e5,
    "lakh": 1e5,
    "lac": 1e5,
    "cr": 1e7,
    "crore": 1e7,
}

# Column names that invite a date reading. Deliberately token-based: "tenure_days"
# must not match, or a duration in days becomes a date in 1900.
DATE_NAME_PATTERN = re.compile(
    r"(^|_)(date|dates|dt|datetime|timestamp|day|on|at|since|until)($|_)",
    re.IGNORECASE,
)

ISO_DATE_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[ T].*)?$")

# 12/03/2024, 12.03.2024, 12-03-2024 — the separator must repeat, which keeps
# decimals ("19.99") and thousands ("1,234") out.
SEPARATED_DATE_PATTERN = re.compile(r"^(\d{1,4})([./-])(\d{1,2})\2(\d{2,4})$")

INTEGER_PATTERN = re.compile(r"^-?\d+$")

# Excel's day 0 is 1899-12-30 (the 1900 leap-year bug is baked into the format).
EXCEL_EPOCH = datetime(1899, 12, 30)

# Serial numbers only get a date reading inside a plausible window: 20000 is
# 1954-09-27 and 60000 is 2064-03-06. A count of anything rarely lands here, and
# when it does the column name has to invite a date reading as well.
EXCEL_SERIAL_RANGE: tuple[int, int] = (20_000, 60_000)

# How much of a column has to parse before the column is called numeric or a
# date. Below this the values are left as text and nothing is discarded.
PARSE_ACCEPTANCE = 0.8

MAX_SAMPLE_FAILURES = 5


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ColumnStats:
    """What happened to one column, in enough detail to argue with.

    ``n_parsed + n_failed + n_missing`` always equals the number of rows kept, so
    a column cannot lose values without the arithmetic showing it.
    """

    name: str
    source_name: str
    kind: str  # "numeric" | "datetime" | "text"
    n_parsed: int
    n_failed: int
    n_missing: int
    sample_failures: tuple[str, ...] = ()
    null_tokens: dict[str, int] = field(default_factory=dict)
    unit: str | None = None  # e.g. "percent" — records that values were rescaled
    currency: str | None = None
    decimal_separator: str | None = None  # "." or "," — decided per column
    # "day-first" | "month-first" | "iso" | "excel-serial"
    date_format: str | None = None
    ambiguous_date_format: bool = False


@dataclass(frozen=True)
class RowFailure:
    """A row that could not be read, kept whole so a person can judge it (I10)."""

    line_number: int
    n_fields: int
    n_expected: int
    raw_line: str
    reason: str


@dataclass(frozen=True)
class ParseResult:
    """Everything the parser decided, alongside what it produced."""

    frame: pd.DataFrame
    columns: dict[str, ColumnStats]
    encoding: str
    delimiter: str
    header_row: int
    preamble: list[str]
    n_rows_in: int
    n_rows_kept: int
    empty_rows_dropped: int
    row_failures: list[RowFailure]
    dropped_columns: list[str]
    renamed_columns: dict[str, str]
    ambiguous_date_columns: list[str]
    warnings: list[str]
    path: str

    @property
    def n_rows_reported(self) -> int:
        """Rows that did not make it into the frame, and were said so out loud."""
        return self.empty_rows_dropped + len(self.row_failures)

    @property
    def requires_disambiguation(self) -> bool:
        """True while a human still has to decide something (I7).

        Callers are expected to refuse to train on a result that says True.
        """
        return bool(self.ambiguous_date_columns)


# ── Encoding ──────────────────────────────────────────────────────────────────


def _detect_encoding(raw: bytes) -> str:
    """Pick an encoding that decodes the whole file, never one that mangles it.

    The order matters. UTF-8 is checked strictly, so a file that is not UTF-8
    fails here rather than arriving as mojibake. cp1252 is preferred over latin-1
    when bytes fall in 0x80-0x9F, because that range is where the two disagree:
    latin-1 maps it to unprintable C1 controls, cp1252 to the curly quotes and
    dashes Excel-on-Windows actually writes.
    """
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        return "utf-8"

    if any(0x80 <= byte <= 0x9F for byte in raw):
        try:
            raw.decode("cp1252")
        except UnicodeDecodeError:
            pass
        else:
            return "cp1252"
    return "latin-1"


# ── Delimiter ─────────────────────────────────────────────────────────────────


def _score_delimiter(
    text: str, delimiter: str, sample_rows: int = 50
) -> tuple[float, int]:
    """Score a candidate by how consistent a table it produces, not by frequency.

    Counting characters picks the comma in a semicolon-delimited file whose
    values contain commas. Consistency of field count across rows does not.
    """
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    counts: list[int] = []
    for row in reader:
        if row:
            counts.append(len(row))
        if len(counts) >= sample_rows:
            break
    if not counts:
        return (0.0, 0)
    modal_count, modal_rows = Counter(counts).most_common(1)[0]
    if modal_count < 2:
        return (0.0, 0)
    return (modal_rows / len(counts), modal_count)


def _detect_delimiter(text: str) -> str:
    best_delimiter = ","
    best_score = (0.0, 0)
    for candidate in CANDIDATE_DELIMITERS:
        score = _score_delimiter(text, candidate)
        if score > best_score:
            best_delimiter, best_score = candidate, score
    return best_delimiter


# ── Records and header ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Record:
    line_number: int  # 1-based, as a person reading the file in an editor counts
    fields: list[str]


def _read_records(text: str, delimiter: str) -> list[_Record]:
    """Read every record, keeping line numbers so failures can be pointed at."""
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    records: list[_Record] = []
    previous_line = 0
    for row in reader:
        # csv.reader reports the line a record ENDS on. A quoted value with a
        # newline in it therefore ends two lines below where a person reading the
        # file sees the row start, and a failure report has to point at the line
        # they will actually look at.
        start_line = previous_line + 1
        previous_line = reader.line_num
        if not row:  # a genuinely blank line carries no fields at all
            continue
        records.append(_Record(line_number=start_line, fields=row))
    return records


def _looks_numeric(value: str) -> bool:
    try:
        float(value.replace(",", ""))
    except ValueError:
        return False
    return True


def _find_header(records: list[_Record]) -> int:
    """Index of the record that is the real header.

    Export banners and blank lines above the header are common enough that
    requiring the operator to pass skiprows would break the promise that an
    uploaded file just works. The header is taken to be the first record whose
    field count matches the shape of the file and whose fields read like names.
    """
    if not records:
        return 0

    modal_count = Counter(len(r.fields) for r in records).most_common(1)[0][0]
    for index, record in enumerate(records):
        if len(record.fields) != modal_count:
            continue
        named = [f.strip() for f in record.fields if f.strip()]
        if len(named) < 2:
            continue
        # A row of measurements is data, not a header.
        if sum(_looks_numeric(f) for f in named) > len(named) / 2:
            continue
        return index
    return 0


def _normalise_headers(
    fields: list[str],
) -> tuple[list[str], dict[str, str], list[int], list[str]]:
    """Strip, name the unnamed, and disambiguate duplicates deterministically.

    Returns the names, a mapping of every renamed column back to the header text
    it came from, the positions that arrived unnamed, and the warnings raised.
    """
    warnings: list[str] = []
    stripped = [f.strip() for f in fields]

    unnamed_positions = [i for i, name in enumerate(stripped) if not name]

    names: list[str] = []
    renamed: dict[str, str] = {}
    seen: Counter[str] = Counter()
    for position, name in enumerate(stripped):
        candidate = name or f"column_{position + 1}"
        seen[candidate] += 1
        if seen[candidate] > 1:
            unique = f"{candidate}_{seen[candidate] - 1}"
            # Two different columns called the same thing: the operator has to
            # be able to tell which is which, and neither may be dropped.
            renamed[unique] = candidate
            names.append(unique)
        else:
            names.append(candidate)

    if renamed:
        warnings.append(
            "duplicate column names in the header: "
            + ", ".join(f"{new} (was a second {old})" for new, old in renamed.items())
            + " — both columns are kept because the parser cannot know which one "
            "you mean"
        )
    return names, renamed, unnamed_positions, warnings


# ── Value cleaning ────────────────────────────────────────────────────────────


def _null_tokens_in(values: list[str]) -> dict[str, int]:
    """Count the null-looking spellings in a column, preserving their raw form."""
    counts: Counter[str] = Counter()
    for value in values:
        if value.lower() in NULL_TOKENS:
            counts[value] += 1
    return dict(counts)


# Thousands separators come in several invisible flavours: an ordinary space, a
# no-break space (U+00A0) and a narrow no-break space (U+202F) are all used to
# group digits, and French and Nordic exports use them routinely.
GROUPING_SPACES = (" ", "\u00a0", "\u202f")


def _strip_grouping(text: str, grouping_character: str) -> str:
    """Remove digit grouping without touching the character that divides."""
    text = text.replace(grouping_character, "")
    for space in GROUPING_SPACES:
        text = text.replace(space, "")
    return text


DECIMAL_COMMA_PATTERN = re.compile(r",\d{1,2}$")


def _decimal_separator(candidates: list[str]) -> str:
    """Decide, for a whole column, whether a comma groups digits or divides them.

    Getting this wrong is a silent tenfold error: strip the comma from the German
    ``1,5`` and it becomes fifteen. As with date order, the decision is made once
    per column from the evidence in it, never per value — a column where some
    commas group and some divide is unreadable by anything downstream.

    Evidence, in order: a value carrying both separators settles it (the rightmost
    one divides), then a comma followed by one or two final digits, which no
    grouping convention produces. Absent both, a comma groups.
    """
    for value in candidates:
        if "," in value and "." in value:
            return "," if value.rindex(",") > value.rindex(".") else "."
    if any(DECIMAL_COMMA_PATTERN.search(value) for value in candidates):
        return ","
    return "."


def _parse_number(
    value: str, decimal_separator: str = "."
) -> tuple[float | None, str | None, str | None]:
    """Parse one value, returning the number, its currency, and its unit.

    Currency symbols, digit grouping (both 1,234,567 and 1,20,000) and a percent
    sign are notation, not data. A percent is rescaled to a fraction here rather
    than downstream, because 45 and 0.45 look equally plausible in a model report
    and only one of them is the number the operator meant.
    """
    text = value.strip()
    currency: str | None = None
    unit: str | None = None

    for symbol in CURRENCY_SYMBOLS:
        if symbol in text:
            currency = symbol
            text = text.replace(symbol, "")

    text = text.strip()
    if text.endswith("%"):
        unit = "percent"
        text = text[:-1].strip()

    multiplier = 1.0
    lowered = text.lower()
    for suffix, factor in MULTIPLIER_SUFFIXES.items():
        if currency is not None and lowered.endswith(suffix):
            stem = text[: -len(suffix)].strip()
            if stem and not stem[-1].isalpha():
                multiplier = factor
                text = stem
                break

    if decimal_separator == ",":
        text = _strip_grouping(text, ".")
        text = text.replace(",", ".")
    else:
        text = _strip_grouping(text, ",")
    if not text:
        return None, currency, unit
    try:
        number = float(text)
    except ValueError:
        return None, currency, unit

    number *= multiplier
    if unit == "percent":
        number /= 100.0
    return number, currency, unit


def _date_components(value: str) -> tuple[int, int, int, str] | None:
    """Split a date-looking string into (first, second, year, style).

    ``style`` is "iso" when the layout settles the order by itself and
    "separated" when it does not — the caller decides the order for the whole
    column from the evidence in it, never per row.
    """
    iso = ISO_DATE_PATTERN.match(value)
    if iso:
        return int(iso.group(2)), int(iso.group(3)), int(iso.group(1)), "iso"

    parts = SEPARATED_DATE_PATTERN.match(value)
    if not parts:
        return None
    first, second, last = int(parts.group(1)), int(parts.group(3)), int(parts.group(4))
    if len(parts.group(1)) == 4:  # YYYY/MM/DD
        return second, last, first, "iso"
    if last < 100:  # two-digit year, windowed the way spreadsheets do
        last += 2000 if last < 70 else 1900
    return first, second, last, "separated"


def _build_date(day: int, month: int, year: int) -> datetime | None:
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


# ── Column parsing ────────────────────────────────────────────────────────────


@dataclass
class _ColumnParse:
    values: list[object]
    kind: str
    n_parsed: int
    n_failed: int
    n_missing: int
    sample_failures: tuple[str, ...] = ()
    null_tokens: dict[str, int] = field(default_factory=dict)
    unit: str | None = None
    currency: str | None = None
    decimal_separator: str | None = None
    date_format: str | None = None
    ambiguous_date_format: bool = False
    warnings: list[str] = field(default_factory=list)


def _as_text(name: str, raw: list[str]) -> _ColumnParse:
    """Leave the column alone. Only blank cells count as missing.

    A text column keeps every spelling it arrived with, including the ones that
    look like nulls, because in a categorical column "None" is frequently the
    operator's word for "no add-on" rather than an absence.
    """
    values: list[object] = [value if value else None for value in raw]
    n_missing = sum(1 for value in values if value is None)
    warnings: list[str] = []

    tokens = _null_tokens_in([v for v in raw if v])
    if tokens:
        warnings.append(
            f"{name}: kept {sum(tokens.values())} value(s) spelled "
            + ", ".join(sorted(tokens))
            + " as categories rather than as missing, because the column is not "
            "numeric and the spelling may be a real category — confirm before "
            "training"
        )

    return _ColumnParse(
        values=values,
        kind="text",
        n_parsed=len(values) - n_missing,
        n_failed=0,
        n_missing=n_missing,
        warnings=warnings,
    )


def _try_excel_serial(name: str, candidates: list[str]) -> str | None:
    """Decide whether a column of integers is really a column of dates.

    Two conditions, both required: the name invites a date reading, and every
    value sits inside a plausible serial window. Either alone turns durations
    and counts into dates in 1900.
    """
    if not DATE_NAME_PATTERN.search(name):
        return None
    low, high = EXCEL_SERIAL_RANGE
    for value in candidates:
        if not INTEGER_PATTERN.match(value) or not low <= int(value) <= high:
            return None
    return "excel-serial"


def _try_dates(
    name: str, raw: list[str], candidates: list[str], dayfirst: bool | None
) -> _ColumnParse | None:
    """Parse a column of dates, or refuse to and say why.

    The order (day-first vs month-first) is settled once for the column from the
    evidence in it: a component above 12 anywhere proves the position it sits in.
    With no evidence the column is returned unparsed and flagged, because a
    per-row or locale guess produces a column where some dates are day-first and
    some are not, which nothing downstream can detect.
    """
    if not candidates:
        return None

    serial = _try_excel_serial(name, candidates)
    if serial:
        return _parse_serial_column(name, raw, serial)

    components: list[tuple[int, int, int, str] | None] = [
        _date_components(value) for value in candidates
    ]
    recognised = [c for c in components if c is not None]
    if len(recognised) < len(candidates) * PARSE_ACCEPTANCE:
        return None
    if not DATE_NAME_PATTERN.search(name) and len(recognised) < len(candidates):
        return None

    separated = [c for c in recognised if c[3] == "separated"]
    if not separated:
        order = "iso"
    elif any(first > 12 for first, _, _, _ in separated):
        order = "day-first"
    elif any(second > 12 for _, second, _, _ in separated):
        order = "month-first"
    elif dayfirst is None:
        return _ambiguous_column(name, raw)
    else:
        order = "day-first" if dayfirst else "month-first"

    return _parse_date_column(name, raw, order)


def _ambiguous_column(name: str, raw: list[str]) -> _ColumnParse:
    parsed = _as_text(name, raw)
    parsed.ambiguous_date_format = True
    parsed.warnings = [
        f"{name}: dates like {raw[0]!r} could be day-first or month-first and "
        "nothing in the column settles it — left unparsed, awaiting your answer"
    ]
    return parsed


def _parse_serial_column(name: str, raw: list[str], date_format: str) -> _ColumnParse:
    tokens = _null_tokens_in(raw)
    values: list[object] = []
    failures: list[str] = []
    n_parsed = n_missing = 0

    for value in raw:
        if not value or value.lower() in NULL_TOKENS:
            values.append(None)
            n_missing += 1
            continue
        try:
            parsed = EXCEL_EPOCH + pd.Timedelta(days=int(value))
        except ValueError:
            values.append(None)
            failures.append(value)
            continue
        values.append(parsed)
        n_parsed += 1

    return _ColumnParse(
        values=values,
        kind="datetime",
        n_parsed=n_parsed,
        n_failed=len(failures),
        n_missing=n_missing,
        sample_failures=tuple(failures[:MAX_SAMPLE_FAILURES]),
        null_tokens=tokens,
        date_format=date_format,
        warnings=[
            f"{name}: read as Excel serial day numbers on the 1899-12-30 origin "
            f"(for example {raw[0]!r} is "
            f"{(EXCEL_EPOCH + pd.Timedelta(days=int(raw[0]))).date()}) — check "
            "this is a date column and not a measurement"
        ]
        if raw and INTEGER_PATTERN.match(raw[0])
        else [f"{name}: read as Excel serial day numbers on the 1899-12-30 origin"],
    )


def _parse_date_column(name: str, raw: list[str], order: str) -> _ColumnParse:
    tokens = _null_tokens_in(raw)
    values: list[object] = []
    failures: list[str] = []
    n_parsed = n_missing = 0

    for value in raw:
        if not value or value.lower() in NULL_TOKENS:
            values.append(None)
            n_missing += 1
            continue
        components = _date_components(value)
        if components is None:
            values.append(None)
            failures.append(value)
            continue
        first, second, year, style = components
        if style == "iso" or order == "month-first":
            month, day = first, second
        else:
            day, month = first, second
        parsed = _build_date(day=day, month=month, year=year)
        if parsed is None:
            values.append(None)
            failures.append(value)
            continue
        values.append(parsed)
        n_parsed += 1

    warnings: list[str] = []
    if failures:
        warnings.append(
            f"{name}: {len(failures)} value(s) did not parse as dates, for example "
            + ", ".join(repr(f) for f in failures[:MAX_SAMPLE_FAILURES])
        )

    return _ColumnParse(
        values=values,
        kind="datetime",
        n_parsed=n_parsed,
        n_failed=len(failures),
        n_missing=n_missing,
        sample_failures=tuple(failures[:MAX_SAMPLE_FAILURES]),
        null_tokens=tokens,
        date_format=order,
        warnings=warnings,
    )


def _try_numeric(
    name: str, raw: list[str], candidates: list[str]
) -> _ColumnParse | None:
    """Parse a column of numbers, including ones wearing currency and percent.

    Returns None when too little of the column parses, in which case it stays
    text — a column that is 30% numeric is not a numeric column with 70% missing
    data, and treating it as one destroys the other 70%.
    """
    if not candidates:
        return None

    separator = _decimal_separator(candidates)
    attempts = [_parse_number(value, separator) for value in candidates]
    parsed_count = sum(1 for number, _, _ in attempts if number is not None)
    if parsed_count < len(candidates) * PARSE_ACCEPTANCE:
        return None

    currencies = {c for _, c, _ in attempts if c}
    percent_values = sum(1 for _, _, u in attempts if u == "percent")
    # A column where only some values carry a percent sign is not a percentage
    # column; rescaling part of it would be the hundredfold error, applied twice.
    unit = "percent" if percent_values == len(candidates) else None

    tokens = _null_tokens_in(raw)
    values: list[object] = []
    failures: list[str] = []
    n_parsed = n_missing = 0

    for value in raw:
        if not value or value.lower() in NULL_TOKENS:
            values.append(None)
            n_missing += 1
            continue
        number, _, value_unit = _parse_number(value, separator)
        if number is None:
            values.append(None)
            failures.append(value)
            continue
        if value_unit == "percent" and unit is None:
            # Kept as a failure rather than silently taken at face value.
            values.append(None)
            failures.append(value)
            continue
        values.append(number)
        n_parsed += 1

    warnings: list[str] = []
    if failures:
        warnings.append(
            f"{name}: {len(failures)} value(s) could not be read as numbers, for "
            "example "
            + ", ".join(repr(f) for f in failures[:MAX_SAMPLE_FAILURES])
            + " — they are recorded as unparsed, not replaced"
        )
    if unit == "percent":
        warnings.append(
            f"{name}: values carried a percent sign and were rescaled to "
            "fractions, so 45% is stored as 0.45"
        )
    if separator == ",":
        warnings.append(
            f"{name}: read the comma as a decimal separator, so 1,5 is one and a "
            "half rather than fifteen — confirm this is what the export meant"
        )
    if len(currencies) > 1:
        warnings.append(
            f"{name}: more than one currency symbol appears ("
            + ", ".join(sorted(currencies))
            + ") — the values are not comparable until you confirm the intent"
        )
    if any(
        currency and _has_multiplier_suffix(value)
        for value, (_, currency, _) in zip(candidates, attempts, strict=True)
    ):
        warnings.append(
            f"{name}: read Indian numbering — a lakh suffix as 100,000 and a "
            "crore suffix as 10,000,000, with digit grouping like 1,20,000 "
            "treated as 120000"
        )

    return _ColumnParse(
        values=values,
        kind="numeric",
        n_parsed=n_parsed,
        n_failed=len(failures),
        n_missing=n_missing,
        sample_failures=tuple(failures[:MAX_SAMPLE_FAILURES]),
        null_tokens=tokens,
        unit=unit,
        currency=sorted(currencies)[0] if len(currencies) == 1 else None,
        decimal_separator=separator,
        warnings=warnings,
    )


def _has_multiplier_suffix(value: str) -> bool:
    stripped = value.strip().lower().rstrip()
    return any(stripped.endswith(suffix) for suffix in MULTIPLIER_SUFFIXES)


def _parse_column(name: str, raw: list[str], dayfirst: bool | None) -> _ColumnParse:
    """Give one column the reading it earns, in the order dates, numbers, text."""
    non_blank = [value for value in raw if value]
    candidates = [value for value in non_blank if value.lower() not in NULL_TOKENS]

    dates = _try_dates(name, raw, candidates, dayfirst)
    if dates is not None:
        return dates

    numbers = _try_numeric(name, raw, candidates)
    if numbers is not None:
        return numbers

    return _as_text(name, raw)


def _series(parsed: _ColumnParse) -> pd.Series:
    if parsed.kind == "numeric":
        return pd.Series(parsed.values, dtype="float64")
    if parsed.kind == "datetime":
        return pd.Series(parsed.values, dtype="datetime64[ns]")
    return pd.Series(parsed.values, dtype="object")


# ── Rows ──────────────────────────────────────────────────────────────────────


def _split_rows(
    records: list[_Record], n_expected: int, lines: list[str]
) -> tuple[list[list[str]], list[RowFailure], int, list[int]]:
    """Sort records into kept rows, reported failures, and empty rows.

    A ragged row is a judgement call the parser is not entitled to make on the
    operator's behalf: trailing empty fields are padding and recoverable, but a
    row with fields missing or extra data in them cannot be aligned without
    guessing which column the values belong to, so it is reported whole.
    """
    kept: list[list[str]] = []
    failures: list[RowFailure] = []
    empty_rows = 0
    trimmed_lines: list[int] = []

    for record in records:
        fields = [f.strip() for f in record.fields]
        raw_line = (
            lines[record.line_number - 1]
            if record.line_number <= len(lines)
            else ""
        )

        if not any(fields):
            empty_rows += 1
            continue

        if len(fields) == n_expected:
            kept.append(fields)
            continue

        if len(fields) > n_expected and not any(fields[n_expected:]):
            kept.append(fields[:n_expected])
            trimmed_lines.append(record.line_number)
            continue

        failures.append(
            RowFailure(
                line_number=record.line_number,
                n_fields=len(fields),
                n_expected=n_expected,
                raw_line=raw_line,
                reason=(
                    f"{len(fields)} fields where the header declares {n_expected}; "
                    "aligning it would mean guessing which columns the values "
                    "belong to"
                ),
            )
        )

    return kept, failures, empty_rows, trimmed_lines


# ── Entry point ───────────────────────────────────────────────────────────────


def read_table(
    path: str | Path,
    *,
    encoding: str | None = None,
    delimiter: str | None = None,
    dayfirst: bool | None = None,
) -> ParseResult:
    """Read a delimited file and report everything that had to be decided.

    Nothing here raises on bad data. A file with three unreadable rows returns
    the six good ones plus a report naming the three, because aborting an upload
    over a fraction of it is the behaviour that makes operators pre-clean their
    data by hand — which is the problem this platform exists to remove.

    ``encoding``, ``delimiter`` and ``dayfirst`` are operator overrides. They are
    the answers to questions the parser asks; ``dayfirst`` in particular resolves
    a column the parser has refused to guess at, and evidence inside a column
    still wins over it, so a column that proves itself day-first is not reordered
    by an operator's blanket answer.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cannot read {path}: no such file")

    raw_bytes = path.read_bytes()
    detected_encoding = encoding or _detect_encoding(raw_bytes)
    text = raw_bytes.decode(detected_encoding)
    lines = text.splitlines()

    warnings: list[str] = []
    if encoding is None and detected_encoding != "utf-8":
        warnings.append(
            f"encoding detected as {detected_encoding} rather than utf-8 — "
            "characters were decoded on that basis"
        )

    detected_delimiter = delimiter or _detect_delimiter(text)
    records = _read_records(text, detected_delimiter)
    if not records:
        raise ValueError(f"cannot read {path}: the file contains no records")

    header_index = _find_header(records)
    header_record = records[header_index]
    preamble = lines[: header_record.line_number - 1]
    if header_index > 0:
        warnings.append(
            f"header row found on line {header_record.line_number}, not line 1; the "
            f"{len(preamble)} line(s) above it were read as a preamble and skipped"
        )

    names, renamed, unnamed_positions, header_warnings = _normalise_headers(
        header_record.fields
    )
    warnings.extend(header_warnings)

    data_records = records[header_index + 1 :]
    kept_rows, row_failures, empty_rows, trimmed_lines = _split_rows(
        data_records, len(names), lines
    )

    if trimmed_lines:
        warnings.append(
            "trailing empty fields were trimmed from line(s) "
            + ", ".join(str(line) for line in trimmed_lines)
            + " so the rows fit the header"
        )
    for failure in row_failures:
        warnings.append(
            f"line {failure.line_number} not read: {failure.reason}. Raw line: "
            f"{failure.raw_line!r}"
        )
    if not kept_rows:
        warnings.append(
            f"no data rows were read: the header on line "
            f"{header_record.line_number} is followed by nothing the parser could "
            "keep"
        )
    if empty_rows:
        warnings.append(
            f"{empty_rows} empty row(s) dropped so the row count reported to you "
            "is the number of real records"
        )

    columns_raw: dict[str, list[str]] = {
        name: [row[position] for row in kept_rows]
        for position, name in enumerate(names)
    }

    dropped_columns: list[str] = []
    for position in reversed(unnamed_positions):
        name = names[position]
        if not any(columns_raw[name]):
            dropped_columns.append(name)
            del columns_raw[name]
            names.pop(position)
    if dropped_columns:
        warnings.append(
            "dropped unnamed, entirely empty column(s) "
            + ", ".join(reversed(dropped_columns))
            + " — usually a trailing delimiter in the header"
        )
    elif unnamed_positions:
        warnings.append(
            "column(s) "
            + ", ".join(names[position] for position in unnamed_positions)
            + " arrived with no header name and were given positional names"
        )

    frame_data: dict[str, pd.Series] = {}
    stats: dict[str, ColumnStats] = {}
    ambiguous: list[str] = []

    for name in names:
        parsed = _parse_column(name, columns_raw[name], dayfirst)
        frame_data[name] = _series(parsed)
        warnings.extend(parsed.warnings)
        if parsed.ambiguous_date_format:
            ambiguous.append(name)
        stats[name] = ColumnStats(
            name=name,
            source_name=renamed.get(name, name),
            kind=parsed.kind,
            n_parsed=parsed.n_parsed,
            n_failed=parsed.n_failed,
            n_missing=parsed.n_missing,
            sample_failures=parsed.sample_failures,
            null_tokens=parsed.null_tokens,
            unit=parsed.unit,
            currency=parsed.currency,
            decimal_separator=parsed.decimal_separator,
            date_format=parsed.date_format,
            ambiguous_date_format=parsed.ambiguous_date_format,
        )

    frame = pd.DataFrame(frame_data, columns=names)

    return ParseResult(
        frame=frame,
        columns=stats,
        encoding=detected_encoding,
        delimiter=detected_delimiter,
        header_row=header_record.line_number - 1,
        preamble=preamble,
        n_rows_in=len(data_records),
        n_rows_kept=len(kept_rows),
        empty_rows_dropped=empty_rows,
        row_failures=row_failures,
        dropped_columns=dropped_columns,
        renamed_columns=renamed,
        ambiguous_date_columns=ambiguous,
        warnings=warnings,
        path=str(path),
    )
