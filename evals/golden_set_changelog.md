# Golden set changelog

## 2026-10-28 — audit of failures (see `failure_analysis.md`)

Five of the 30 rows changed. The other 25 are untouched. The pre-change file is in git history (commit before this change).

| ID | Change | Why |
|---|---|---|
| SVC-04 | `expected_sections` `["10.5","3.8"]` -> `["10.5"]`; `3.8` moved to `related_sections`; `source_note` explains | §3.8 is only cross-referenced by §10.5 and its text is about Prepaid Transportation Credits; the answer is complete from §10.5 alone |
| LIC-05 | `expected_sections` `["2.1","2.3","2.5"]` -> `["2.1"]`; `2.3`, `2.5` moved to `related_sections` | the answer, including the five-year minimum, is entirely in §2.1; §2.3 and §2.5 only support it |
| LIC-04 | `expected_sections` `["4.4","7.5"]` -> `["4.4"]`; `7.5` moved to `related_sections` | §7.5 (assignment) is a separate point; the change-of-control answer is §4.4 |
| SUP-06 | `expected_answer` reworded | the old text said termination is "only" for breach or bankruptcy, but §4.3 opens "Without limiting the other rights to terminate set forth in this Agreement"; the new text says no convenience right appears and names §4.3 |
| SUP-06, DIS-06 | new field `acceptable_alternative` | both are answerable by omission: a `found: true` answer saying "no, only for cause" / "no, but §15.B lets Ingram terminate anyway" is defensible. Not used by the scripts yet; it is for the judge step |

Not changed (noted in `failure_analysis.md`): SVC-05 `source_span` sits in §3.6 while §3.5 is listed first (metadata only); DIS-05 keeps expected `16` (the exclusion paragraph is unlettered in the source); JV rows keep heading numbers although the contract's own cross-references are off by one.

## Effect on saved runs (`scripts/rescore.py`, no model calls)

Only SVC-04 changes label (`partial` -> `ok`). Top 8 (`k8-postfix`): "all golden clauses" retrieval and citation 92% -> 96%; top 5: 80% -> 84%. The "any golden clause" numbers and the unanswerable result do not change. LIC-05 remains the one failure. The original `summary.json` files are untouched; rescored numbers are in `summary_rescored.json` per run.
