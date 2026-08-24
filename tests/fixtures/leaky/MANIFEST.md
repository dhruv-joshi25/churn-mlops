# Leaky fixture corpus

Seven CSVs for S4 — schema inference and leakage detection. Like the nasty
corpus, they exist **before** the detector, so the detector cannot be written to
only catch what its author happened to think of. This file is the specification.

Each entry states what is planted, which rule must catch it, and — the part that
matters — **which rules must NOT fire**, because a detector that flags
everything is the same as no detector once the operator learns to click through
its warnings.

No Telco column names anywhere, and no two fixtures share a schema (I11).

## Rules that apply to all seven

- Each file parses cleanly through `churnkit.ingest.read_table` first. Leakage
  detection is a layer on the parser, not a second parser.
- A finding names the column, the rule, and the evidence that tripped it (I10).
- Findings are a **proposal**. Nothing here may reach training without explicit
  human confirmation, and a blocking finding needs typed confirmation (I5, I7).

---

## 01_cancellation_date.csv — the leak S4 names explicitly

**Target:** `did_leave`. **Planted:** `cancellation_date`, populated only on
churned rows.

**Must catch:** two independent rules — the `cancel*` name pattern, and
null-mask correlation (r = 1.000 against the target). Either alone is enough;
both firing is the expected result and the finding should say so.

**Severity:** blocking. This is the archetypal churn leak — the date exists
*because* the customer left, so it cannot be known at prediction time.

**Must not:** flag `months_active` or `monthly_spend`, which are ordinary
predictive features.

## 02_perfect_predictor.csv — a perfect separator with an innocent name

**Target:** `lapsed`. **Planted:** `engagement_index`, AUC = 1.000 against the
target on its own. Retained rows score 70–98, lapsed rows 3–27, no overlap.

**Must catch:** the single-column AUC rule, and *only* that rule. Nothing about
the name "engagement_index" invites suspicion, which is the point — this fixture
proves the statistical test does real work rather than dressing up a keyword
list.

**Severity:** blocking.

**Must not:** be caught by name pattern. If it is, the pattern list is too wide.

## 03_null_mask_leak.csv — the values are noise, the missingness is the leak

**Target:** `attrited`. **Planted:** `followup_notes_len`, present for churned
rows and absent for retained ones. The values themselves carry no signal.

**Must catch:** null-mask correlation (r = 1.000). A detector that only inspects
parsed values sees noise here and passes the file — that is the failure this
fixture exists to catch.

**Severity:** blocking.

## 04_post_window_dates.csv — knowing the future

**Target:** `not_renewed`. **Observation window ends 2024-06-30.** `last_seen_on`
holds August 2024 dates for the non-renewing rows.

**Must catch:** the datetime-after-window rule, which requires the caller to
supply the window end. With no window supplied this file has no detectable leak
and the detector must say the check was **skipped**, not that it passed — an
unrun check reported as a pass is exactly the silent-success failure the nasty
corpus was built around.

**Must not:** trip the AUC or null-mask rules. Verified: it trips neither.
`signup_on` is legitimately before the window and must stay unflagged.

**Severity:** blocking when the window is known.

## 05_constant_within_class.csv — separation without a number

**Target:** `stopped`. **Planted:** `retention_code`, `R-KEEP` on every retained
row and `R-LOST` on every churned one.

**Must catch:** the constant-within-class rule. The column is categorical, so
the AUC rule does not see it — this fixture is why that rule alone is not
enough.

**Severity:** blocking.

**Must not:** flag `billing_mode` or `city`, which vary within both classes.

## 06_clean_baseline.csv — the false-positive control

**Target:** `left_service`. **Planted: nothing.** Tenure, monthly value and
ticket counts are genuinely correlated with churn, with overlapping
distributions and no separating column.

**Must produce ZERO blocking findings.** This is the most important fixture in
the corpus. Every threshold in the detector is a trade, and this file is the
only thing measuring what the trade costs. A detector that catches all six leaks
above and also flags this one is not usable: the operator learns that the
warnings are noise and overrides the real one when it comes.

Verified against the shipped thresholds: no column reaches 0.95 AUC, no
null-mask correlation exceeds 0.9, no column is constant within a class.

## 07_roles.csv — one column per role, for Part A

**Target candidate:** `is_gone`. Every other column exercises exactly one role:

| Column | Expected role |
| --- | --- |
| `row_uuid` | identifier — unique across all rows |
| `country_code` | categorical, low cardinality — 4 values |
| `free_note` | free text — long, near-unique, sentence-shaped |
| `opened_at` | datetime |
| `score` | numeric, continuous |
| `tenant_label` | constant — one value throughout |
| `sku_reference` | categorical, high cardinality — near-unique short codes |
| `is_gone` | target candidate — binary, two values |

**Must not:** call `sku_reference` an identifier merely because it is
high-cardinality, or `row_uuid` a categorical merely because it is text. The
distinction is uniqueness and name shape, and confidence should be lower where
the two roles genuinely overlap.

## 08_non_english.csv — a dataset with no English in it

**Target:** `gekündigt`. A German gym's membership export: semicolon-delimited,
a three-line preamble, German decimal commas (`59,90`), dotted day-first dates
(`01.02.2023`), umlauts in the data and in a column name, `ja`/`nein` instead of
yes/no, and `mitglied_nr` instead of any recognisable id name.

**Planted: nothing.** This fixture holds no leak. It is here because the
platform's promise is that *any* company points it at *their* data, and an
earlier version of the inference layer found neither a target nor an identifier
in this file — it recognised `{0,1}`, `{yes,no}` and `{true,false}` and nothing
else, so a company that does not operate in English got a proposal with every
important field blank.

**Must catch:** `gekündigt` as the target and `mitglied_nr` as the identifier.
The target is found because it is the only column with exactly two values, and
the identifier because it is unique on every row and sits first — neither of
which depends on the operator's language.

**Must report honestly:** the target confidence must fall below 0.6 and a
warning must say the choice was a guess. Getting the right answer for a weak
reason is still a weak reason, and I7 gives the human the final say precisely so
that low confidence can be surfaced rather than hidden.

**Must not:** guess which of `ja`/`nein` means the customer left without saying
so. The proposal names the column; the operator confirms the direction.
