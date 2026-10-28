"""Recall@20 check: where does the golden clause rank for each method?

    .venv/bin/python scripts/retrieval_check.py
"""
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.chunker import section_matches  # noqa: E402
from legal_rag.retrieval import search_bm25, search_embeddings  # noqa: E402

K = 20
METHODS = {"dense": search_embeddings, "bm25": search_bm25}


def norm(s):
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", s).lower().strip()


def first_rank(results, pred):
    return next((i for i, (c, _) in enumerate(results, 1) if pred(c)), None)


def fmt(r):
    return "-" if r is None else str(r)


def main():
    meta = json.loads((ROOT / "data" / "contracts_meta.json").read_text())
    cid_by_file = {stem: m["contract_id"] for stem, m in meta.items()}
    rows = [json.loads(l) for l in open(ROOT / "golden_set.jsonl")]
    rows = [r for r in rows if r["expected_sections"]]

    report, stats = [], {m: {"sec": [], "all": [], "span": []} for m in METHODS}
    print(f"{'id':7}{'type':11}{'expected':22}" + "".join(f"| {m}: sec(all) span " for m in METHODS))
    for g in rows:
        cid = cid_by_file[g["cuad_file"]]
        parts = [norm(p).strip(" .") for p in re.split(r"\s*(?:\.\.\.|…)\s*", g["source_span"]) if p.strip()]
        entry = {"id": g["id"], "type": g["type"], "expected": g["expected_sections"]}
        line = f"{g['id']:7}{g['type']:11}{','.join(g['expected_sections'])[:20]:22}"
        for name, fn in METHODS.items():
            res = fn(cid, g["question"], K)
            per_sec = [first_rank(res, lambda c, s=s: section_matches(s, c["clause_ids"])) for s in g["expected_sections"]]
            sec = min((r for r in per_sec if r), default=None)
            allr = None if any(r is None for r in per_sec) else max(per_sec)
            span = first_rank(res, lambda c: all(p in norm(c["body"]) for p in parts))
            stats[name]["sec"].append(sec)
            stats[name]["all"].append(allr)
            stats[name]["span"].append(span)
            entry[name] = {"section_rank": sec, "all_sections_rank": allr, "span_rank": span,
                           "top5": [c["chunk_id"] for c, _ in res[:5]]}
            line += f"| {fmt(sec):>3}({fmt(allr):>3})   {fmt(span):>3}      "
        report.append(entry)
        print(line)

    n = len(rows)
    print(f"\n== summary over {n} questions (k={K}) ==")
    for name, s in stats.items():
        def pct(xs, cut=K):
            return sum(1 for x in xs if x is not None and x <= cut) / n
        found = [x for x in s["sec"] if x]
        print(f"{name:6} section@20={pct(s['sec']):.2f}  all-sections@20={pct(s['all']):.2f}  "
              f"span@20={pct(s['span']):.2f} | rank1={sum(x == 1 for x in s['sec'])}  "
              f"within5={sum(x is not None and x <= 5 for x in s['sec'])}  "
              f"median rank (found)={statistics.median(found) if found else '-'}  "
              f"misses={[rows[i]['id'] for i, x in enumerate(s['sec']) if x is None]}")

    (ROOT / "evals" / "retrieval_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
