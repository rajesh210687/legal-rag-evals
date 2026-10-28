"""Re-score saved runs against the current golden_set.jsonl without calling the model.

    .venv/bin/python scripts/rescore.py k8-postfix k5 ...

Reads evals/runs/<tag>/results.jsonl, recomputes the golden-dependent fields (ranks kept for
clauses still expected, in-prompt, citation match, label) and writes summary_rescored.json
next to the original summary.json, which is left untouched.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_eval  # noqa: E402  (reuses label() and summarize())
from legal_rag.chunker import section_matches  # noqa: E402
from legal_rag.evaluation import covered, first_and_worst  # noqa: E402


def rescore(tag):
    golden = {g["id"]: g for g in map(json.loads, open(ROOT / "golden_set.jsonl"))}
    meta = json.loads((ROOT / "data" / "contracts_meta.json").read_text())
    cid = {k: v["contract_id"] for k, v in meta.items()}
    chunks = {c: {x["chunk_id"].split(":", 1)[1]: x for x in map(json.loads, open(ROOT / "data" / "chunks" / f"{c}.jsonl"))}
              for c in cid.values()}
    rows = [json.loads(l) for l in open(ROOT / "evals" / "runs" / tag / "results.jsonl")]
    out, changed = [], {}
    for r in rows:
        g = golden[r["id"]]
        old_label = r["label"]
        expected = g["expected_sections"]
        r = dict(r, expected_sections=expected)
        by_id = chunks[cid[g["cuad_file"]]]
        if expected:
            ranks = {m: {s: r["ranks"][m][s] for s in expected} for m in r["ranks"]}
            r["ranks"], r["rank_first_worst"] = ranks, {m: first_and_worst(v) for m, v in ranks.items()}
            sent = [by_id[e] for e in r["excerpt_ids"]]
            r["golden_in_prompt_any"] = bool(covered(sent, expected))
            r["golden_in_prompt_all"] = len(covered(sent, expected)) == len(expected)
            cited = [by_id[c] for c in (r["answer"]["citations"] if r["answer"] else []) if c in by_id]
            cov = covered(cited, expected)
            r["cited_golden"], r["cites_golden_any"], r["cites_golden_all"] = cov, bool(cov), len(cov) == len(expected)
        r["label"] = run_eval.label(r, bool(expected))
        if r["label"] != old_label:
            changed[r["id"]] = (old_label, r["label"])
        out.append(r)
    return run_eval.summarize(out), changed


def main():
    for tag in sys.argv[1:] or ["k8-postfix"]:
        new, changed = rescore(tag)
        old = json.loads((ROOT / "evals" / "runs" / tag / "summary.json").read_text())
        (ROOT / "evals" / "runs" / tag / "summary_rescored.json").write_text(json.dumps(new, indent=2))
        print(f"\n== {tag} (current golden set) ==")
        for key, label in (("retrieval_recall_at_k_any", "Retrieval recall (any golden clause)"),
                           ("retrieval_recall_at_k_all", "Retrieval recall (all golden clauses)"),
                           ("citation_accuracy_any", "Citation accuracy (any)"),
                           ("citation_accuracy_all", "Citation accuracy (all)"),
                           ("citation_accuracy_any_given_retrieved", "Citation accuracy given retrieved"),
                           ("unanswerable_found_false", "Unanswerable marked found:false")):
            print(f"  {label:40} {old[key]:5.0f}% -> {new[key]:5.0f}%")
        print(f"  labels {old['labels']} -> {new['labels']}")
        print(f"  label changes: {changed or 'none'}")


if __name__ == "__main__":
    main()
