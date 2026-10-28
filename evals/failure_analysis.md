# Failure analysis

Baseline: `evals/runs/k8-postfix` (hybrid search, top 8 excerpts, Sonnet 5.5). History column: `evals/runs/k5`.
Labels come from per-stage ranks in `results.jsonl` plus a hand review of every non-OK and every fragile pass.
There is no reranking stage (the reranker was dropped), so "lost at reranking" is replaced by "lost at the cut-off".

## Definitions (fixed before labeling)

| Category | Rule |
|---|---|
| Missed by both searches | golden clause ranks beyond 8 in BM25 and in embeddings |
| Lost at fusion | within the top 8 in one search, beyond 8 after fusion |
| Lost at the cut-off | fused rank is past the number of excerpts sent (k) but the clause was found |
| Chunking error | the answer text is split, or sits under the wrong clause, or is buried in a multi-topic chunk |
| Generation failure | golden clause was in the prompt; wrong citation, over-refusal, hallucination, or partial answer |
| Golden-set error | the label is wrong, over-demanding or ambiguous |

## 1. Failures in the baseline (top 8)

| ID | Category | Note |
|---|---|---|
| LIC-05 | Retrieval: missed by both searches (chunking contributor) | §2.1 ranks 16 in BM25, 15 in embeddings, 14 fused. The answer is one sentence 36% of the way into a 244-word chunk that mixes the licence grant, post-termination rights and ownership, and the question ("after the agreement ends…") reads like the termination clauses. The model correctly said "Not found" given what it was sent, so this is not an over-refusal. |
| SVC-04 | Golden-set error (not a real failure) | The answer is correct and complete from §10.5. The "missing" §3.8 is only cross-referenced there; its text is about Prepaid Transportation Credits and contains no Unsubscribed Capacity process. The run labels it `partial` only because §3.8 was in the golden list. |

## 2. What failed at top 5 and what top 8 fixed

| ID | Category at top 5 | Note |
|---|---|---|
| DIS-05 | Lost at the cut-off | Clause 16 fused rank 8. The liability-exclusion paragraph is unlettered in the source and the chunker files it under `16.B` (now its own chunk `16.B#2`). Fixed by k=8. |
| SVC-05 | Lost at the cut-off | §3.6 (the credits) fused rank 8. Top 5 gave a correct but incomplete answer and the model said so honestly. Fixed by k=8. |
| SUP-04 | Lost at the cut-off | §3.2 fused rank 8 (embeddings rank 17, found by BM25). Fixed by k=8. |
| LIC-05 | Missed by both searches | unchanged at top 8 |
| SVC-04 | Golden-set error | unchanged at top 8 |

## 3. Fragile passes (pass today, would fail with a small change)

| ID | Why fragile |
|---|---|
| DIS-05, SVC-05, SUP-04 | the golden clause sits at fused rank exactly 8, so they break at k=7 |
| JV-01, JV-03 | BM25 cannot find them (ranks 53 and 29); embeddings rescue them (vocabulary gap: "date" vs signature block, "last" vs "DURATION") |
| LIC-04 (§7.5), JV-04, SVC-04 (§10.5) | found by embeddings only (BM25 rank 10 to 16) |
| SUP-04 (§3.2) | the one case where BM25 rescues embeddings (rank 3 vs 17) |

## 4. Counts (baseline, top 8, 30 questions)

| Category | Count |
|---|---|
| Retrieval: missed by both | 1 (LIC-05) |
| Retrieval: lost at fusion | 0 |
| Retrieval: lost at the cut-off | 0 (3 at top 5) |
| Chunking contributor | 2 (LIC-05 multi-topic chunk; DIS-05 unlettered paragraph) |
| Generation: wrong citation / over-refusal / hallucination / partial | 0 / 0 / 0 / 0 |
| Golden-set error behind a failure label | 1 (SVC-04) |
| Fully correct (answerable 24 of 25, unanswerable 5 of 5) | 29 of 30 questions answered acceptably |

Generation evidence: when the golden clause reached the prompt, the model cited it every time (100% in all runs); all five unanswerable questions got the exact "Not found" wording with no citation; a screen of every number, year and quoted term in the 24 `found: true` answers against the sent excerpts found nothing unsupported.

## 5. Golden-set audit (rows 1-5 applied; see `golden_set_changelog.md`)

I read the contract text for the clauses behind LIC-04, LIC-05, DIS-04, DIS-05, SVC-04, SUP-06, JV-05 and searched the full text of all five unanswerable questions. For the rest I relied on the model's answer agreeing with the golden answer, which is weaker evidence.

| # | ID | Finding | Change |
|---|---|---|---|
| 1 | SVC-04 | §3.8 is not needed to answer (see above) | move `3.8` from `expected_sections` to `related_sections` |
| 2 | LIC-05 | the answer, including the five-year minimum, is entirely in §2.1; §2.3 and §2.5 only support it | `expected_sections: ["2.1"]`, `related_sections: ["2.3","2.5"]` (the retrieval failure remains) |
| 3 | SUP-06 | golden says termination is "only" for breach or bankruptcy, but §4.3 begins "Without limiting the other rights to terminate set forth in this Agreement" | reword: no convenience-termination right appears; §4.3 gives breach (60 days) and bankruptcy |
| 4 | LIC-04 | §7.5 (assignment) is a separate "Separately…" point, not needed for the change-of-control answer | optional: move `7.5` to `related_sections` |
| 5 | DIS-06, SUP-06 | "unanswerable" by omission: a defensible "No, but §15.B lets Ingram terminate anyway" would fail the `found: false` metric | keep as unanswerable; record an acceptable alternative for the judge block |
| 6 | SVC-05 (not applied) | `source_span` is in §3.6 while §3.5 is listed first | metadata only; add a span for §3.5 or leave |
| 7 | DIS-05 (not applied) | the exclusion paragraph is unlettered in the source | no change to the golden set; the chunker choice is noted above |
| 8 | JV-05 and other JV rows (not applied) | the contract's internal cross-references are off by one against its own headings (it calls IP "clause 5", its heading is 6) | none; the golden set uses heading numbers, which is correct |

Real corrections: 3 (rows 1, 2, 3), applied, plus the optional row 4 and the row 5 notes. After rescoring, only SVC-04 changes label (`partial` -> `ok`); LIC-05 is still a retrieval failure, so the baseline is now 29 ok and 1 retrieval miss. All five unanswerable questions are genuinely absent from the contract text.

## 6. Caveats
- One reviewer, 30 questions: counts are counts, not rates.
- The three headline metrics do not check answer correctness; this file's correctness claims rest on my reading of all 30 answers.
- Model wording varies between identical runs (15 of 30 answers differed in wording between two top-5 runs); the labels did not change.
