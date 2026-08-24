# Nasty-CSV fixture corpus

Fifteen deliberately awful CSVs, one failure mode each. They exist **before** the
parser so the parser cannot be written to only handle what its author thought of.
This file is the specification of correct parsing — a fixture without a manifest
entry proves nothing.

Every entry states three things: what is wrong with the file, what a correct
parse produces, and what the parser must **never** do silently. That last one is
what matters: `pandas.read_csv` will "succeed" on most of these files while
destroying the data. Silent success is the failure mode this corpus exists to
catch.

Column names are deliberately different in every file. No two fixtures share a
schema, and none of them use Telco's column names (I11). A parser that passes
this corpus has not been tuned to one dataset.

## Rules that apply to all fifteen

- No `dropna`, no `fillna`, no silent coercion anywhere in the parse path.
- A value that cannot be parsed is **counted and reported** with a sample of the
  offending raw values, never quietly replaced (I10 — the report names the
  column).
- Detected encoding, detected delimiter, detected header row, and every warning
  are returned as part of `ParseResult`, not logged and forgotten.
- Row counts must reconcile: `n_rows_in == n_rows_kept + n_rows_reported`.

---

## 01_semicolon.csv — delimiter is not a comma

**Wrong:** semicolon-delimited, and the `plan_name` values contain commas
(`Pro, annual, discounted`), so a sniffer that picks the most frequent character
picks `,` and is wrong.

**Correct parse:** delimiter `;`, 4 columns
(`account_ref`, `plan_name`, `monthly_fee`, `left_us`), 8 rows. `monthly_fee`
numeric. `plan_name` keeps its internal commas intact.

**Must not silently:** split on `,` and produce ragged 5+ column output; pick the
frequency-winning delimiter without checking that it yields a consistent field
count across rows.

## 02_tab_quoted.tsv — tabs inside quoted fields

**Wrong:** tab-delimited, and quoted `notes` values contain literal tabs
(`"called support<TAB>twice<TAB>in March"`).

**Correct parse:** delimiter tab, 4 columns, 6 rows, quoting respected — tabs
inside quotes are data, not separators. M01's `notes` is one value.

**Must not silently:** treat quoted tabs as delimiters and shift the remaining
columns left; strip the embedded tabs to make the row fit.

## 03_utf8_bom.csv — byte-order mark

**Wrong:** file starts with `EF BB BF`. Verified behaviour: pandas' C engine
strips the BOM, but a raw `open(path, encoding='utf-8')` or `csv.reader` — which
is exactly what a delimiter sniffer and a header-row detector read from — returns
`'\ufeffcustomer_key'` as the first field. So the bug appears in the detection
layer, not in `read_csv`, and it appears as a column name that looks correct in a
terminal and does not compare equal to `customer_key`.

**Correct parse:** encoding reported as `utf-8-sig`, first column named
`customer_key`, 4 columns, 6 rows.

**Must not silently:** carry the BOM into the column name — downstream that
becomes a schema mapping that cannot match `customer_key`, and a `KeyError`
raised far from its cause.

## 04_latin1.csv — Latin-1, not UTF-8

**Wrong:** accented names encoded latin-1; byte `0xE9` (`é`) at offset 51 is an
invalid UTF-8 continuation byte, so a UTF-8 read raises.

**Correct parse:** encoding detected as latin-1 (or cp1252), 5 rows,
`full_name` values readable — `José Álvarez`, `Renée Dupré`, `Jürgen Müller`,
`Sofía Núñez`, `Anaïs Béranger`.

**Must not silently:** open with `errors='replace'` and produce `Jos?` mojibake,
or skip the offending rows. Both corrupt customer identifiers.

## 05_cp1252_smart_quotes.csv — Windows-1252 punctuation

**Wrong:** curly quotes (`0x91`/`0x92`/`0x93`/`0x94`) and en dashes (`0x96`) from
Excel-on-Windows. `0x93` at offset 49 is an invalid UTF-8 start byte. Distinct
from fixture 04: cp1252 and latin-1 disagree in exactly the `0x80–0x9F` range, so
guessing latin-1 here yields control characters instead of quotes.

**Correct parse:** encoding cp1252, 5 rows, `last_note` reads
`"price is too high"` with curly quotes preserved, `segment` reads `Mid–Market`
with an en dash.

**Must not silently:** decode as latin-1 and leave C1 control characters in the
text; strip non-ASCII bytes.

## 06_dates_dotted.csv — DD.MM.YYYY

**Wrong:** dotted European dates. `25.12.2023` and `30.06.2023` have a first
component > 12, which proves day-first order for the whole column.

**Correct parse:** `signup_date` and `last_seen_date` parsed day-first. U1 signup
= 2023-03-07, U1 last_seen = 2023-12-25, U3 signup = 2023-06-30. 6 rows.

**Must not silently:** fall back to month-first for the rows where both
components are ≤ 12 (`07.03.2023`, `11.11.2022`). A per-row guess produces a
column where some dates are day-first and some month-first — worse than failing,
because nothing downstream can detect it.

## 07_dates_excel_serial.csv — dates as integers

**Wrong:** `start_date` and `renewal_date` are Excel serial numbers. The file also
contains genuine integer measures (`claims_count`, `tenure_days`) of overlapping
magnitude, so "integers in a date-ish range" is not a safe rule.

**Correct parse:** serials interpreted on the 1899-12-30 origin. P1 start =
2023-01-01, P1 renewal = 2024-01-01, P2 start = 2022-01-01, P4 start =
2020-01-01, P6 start = 2021-01-01. `claims_count` and `tenure_days` stay numeric.
6 rows.

**Must not silently:** convert every integer column in the serial range to a date
(`tenure_days` = 365 would become 1900-12-30), or leave `start_date` numeric so
the model trains on 44927 as a magnitude. Either way the parser states in its
warnings which columns it read as serials.

## 08_dates_ambiguous.csv — genuinely undecidable

**Wrong:** every component of every date is ≤ 12 (`03/04/2024`, `05/06/2024`).
There is no evidence in the file for day-first or month-first.

**Correct parse:** **do not guess.** Return the columns with an
`ambiguous_date_format` flag naming `created_on` and `closed_on`, requiring
disambiguation from the operator before use.

**Must not silently:** apply the locale default, `dayfirst=False`, or a format
inferred from a different file. This fixture tests I7 most directly — inference
proposes, a human decides.

## 09_null_spellings.csv — nine spellings of missing, one of which is data

**Wrong:** `support_calls` and `last_payment` use empty, `NA`, `N/A`, `null`,
`NULL`, `-`, `--`, `?` interchangeably. The trap: `addon_tier` has the literal
value `None`, which is a **real category** meaning "no add-on", in 4 of 10 rows.

**Correct parse:** 10 rows. The null spellings in the numeric columns become
missing, counted per column. `addon_tier` keeps `None` as a category with 4
occurrences and 0 missing.

**Verified default behaviour:** `pandas.read_csv` treats `None` as missing (it is
in `STR_NA_VALUES`), so 4 of 10 `addon_tier` values silently become NaN. In the
other direction `-`, `--` and `?` are *not* pandas defaults, so they survive as
strings and hold `support_calls` at object dtype — the column then looks
categorical to any role inference that trusts dtype.

**Must not silently:** apply a global null-token list across every column and
delete the `None` category — that converts 40% of the add-on tiers into missing
data and biases everything the model learns about it. Null-token handling is
per-column and reported, never global and invisible.

## 10_currency_numerics.csv — numbers wearing costumes

**Wrong:** `order_value` is `$1,234.56`, `discount_rate` is `45%`,
`lifetime_value` mixes `₹1.2L` (Indian lakh = 120,000), `₹45,000`, and
`₹1,20,000` (Indian digit grouping, not thousands).

**Correct parse:** `order_value` → 1234.56, 98.00, 12000.00, 450.25, 7890.10,
64.99, currency recorded as metadata. `discount_rate` → 0.45, 0.05, 0.00, 0.125,
0.30, 1.00, percent unit recorded so the scaling is visible. `lifetime_value` is
the hard one: `₹45,000` → 45000 and `₹1,20,000` → 120000 are recoverable;
`₹1.2L`, `₹8.5L`, `₹2.4L` are recoverable only if the parser understands lakh —
if it does not, they are **reported as unparsed with the raw sample**, not
coerced.

**Must not silently:** turn `45%` into 45.0 (a hundredfold error that looks
entirely plausible in a model report), turn `₹1.2L` into 1.2, or drop the rows it
cannot parse.

## 11_duplicate_columns.csv — the same header twice

**Wrong:** `amount` appears twice with **different** values; `region` appears
twice with identical values. 6 header fields, 4 distinct names.

**Correct parse:** deterministic disambiguation — `amount`, `amount_1`, `region`,
`region_1`, or an equivalent documented scheme applied identically on every run.
Both `amount` columns survive with their own data: X1 = 100.00 and 250.00. A
warning names every renamed column.

**Must not silently:** keep the last occurrence and drop the first (older pandas
behaviour loses `amount` = 100.00 entirely); collapse the identical duplicates —
the parser cannot know which `amount` the operator means, so it surfaces both and
lets them choose.

## 12_header_row_three.csv — junk above the header

**Wrong:** two lines of export banner and one blank line precede the real header
on line 4 (index 3). Read naively, the single column name becomes
`Quarterly churn export - CONFIDENTIAL`.

**Correct parse:** header row detected at index 3; columns `partner_code`,
`contract_value`, `months_active`, `renewed`; 5 rows; `contract_value` numeric.
The skipped preamble is reported, not discarded.

**Must not silently:** use row 0 as the header, or require the operator to pass
`skiprows=3` by hand — the platform promise is that an uploaded CSV works without
code.

## 13_trailing_empty.csv — padding rows and a ghost column

**Wrong:** a trailing empty column created by a line-ending comma in the header,
plus three all-empty rows and one genuinely blank line at the end. Excel exports
look exactly like this.

**Correct parse:** 4 real data rows (W1–W4), 4 real columns. The unnamed empty
trailing column is dropped **with a warning**; the empty rows are dropped **with
a count**.

**Must not silently:** produce an `Unnamed: 4` all-NaN column that then becomes a
feature; count the empty rows toward the row total reported to the operator,
making the dataset look larger than it is.

## 14_whitespace_padded.csv — padding everywhere

**Wrong:** headers padded with spaces and tabs (`  tenant_id `, tab-wrapped
`plan_code`), values padded (`  BASIC  `), and two cells containing **only**
whitespace (T3's `plan_code`, T5's `months_billed`).

**Correct parse:** headers stripped to `tenant_id`, `plan_code`, `months_billed`,
`is_gone`. Values stripped, so `BASIC` and `  BASIC  ` are one category, not two.
Whitespace-only cells become missing and are counted. 5 rows.

**Must not silently:** leave the padding so `BASIC` and `BASIC ` encode as two
distinct categories — this inflates cardinality and produces one-hot columns that
can never match at serve time.

## 15_ragged_rows.csv — inconsistent field counts

**Wrong:** the header declares 4 fields. DV2 has 3, DV3 has 5, DV5 has 1, DV6 has
6 (two trailing empties), one row is `,,,`, the rest have 4.

**Correct parse:** every ragged row reported by line number with its field count
and the raw line as the sample. DV6's trailing empties are recoverable (4 real
values plus padding) and may be kept with a warning; DV2, DV3 and DV5 cannot be
resolved without guessing and are reported as failures. The `,,,` row is an empty
row, handled as in fixture 13. Whatever the parser decides,
`n_rows_in == n_rows_kept + n_rows_reported` must hold.

**Must not silently:** raise `ParserError` and abort the whole upload over 3 bad
lines out of 9; or use `on_bad_lines='skip'` and hand back a clean-looking frame
with rows missing and no record of what disappeared.
