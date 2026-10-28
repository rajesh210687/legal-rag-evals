"""Sweep RRF constant and dense weight on the golden set.

    .venv/bin/python scripts/rrf_sweep.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.chunker import section_matches  # noqa: E402
from legal_rag.retrieval import search_bm25, search_embeddings, search_hybrid  # noqa: E402

SHOW = ["LIC-05", "DIS-05", "JV-01", "JV-03", "LIC-02", "SVC-04"]


def rank_of(res, expected):
    per = [next((i for i, (c, _) in enumerate(res, 1) if section_matches(s, c["clause_ids"])), None) for s in expected]
    first = min((r for r in per if r), default=None)
    return first, (None if any(r is None for r in per) else max(per))


def main():
    meta = json.loads((ROOT / "data" / "contracts_meta.json").read_text())
    cid = {k: v["contract_id"] for k, v in meta.items()}
    rows = [json.loads(l) for l in open(ROOT / "golden_set.jsonl")]
    rows = [r for r in rows if r["expected_sections"]]

    configs = {"dense": lambda c, q: search_embeddings(c, q, 20), "bm25": lambda c, q: search_bm25(c, q, 20)}
    for rk in (60, 20, 10):
        for w in (1.0, 2.0):
            configs[f"rrf k={rk} w={w:g}"] = lambda c, q, rk=rk, w=w: search_hybrid(c, q, 20, rrf_k=rk, dense_weight=w)

    print(f"{'config':16}{'any@20':>7}{'all@20':>7}{'rank1':>6}{'top5':>5}{'top3':>5}{'allTop5':>8}   " + " ".join(f"{s:>6}" for s in SHOW))
    for name, fn in configs.items():
        firsts, alls = {}, {}
        for g in rows:
            firsts[g["id"]], alls[g["id"]] = rank_of(fn(cid[g["cuad_file"]], g["question"]), g["expected_sections"])
        f = list(firsts.values())
        a = list(alls.values())
        print(f"{name:16}{sum(x is not None for x in f):>7}{sum(x is not None for x in a):>7}"
              f"{sum(x == 1 for x in f):>6}{sum(x is not None and x <= 5 for x in f):>5}"
              f"{sum(x is not None and x <= 3 for x in f):>5}{sum(x is not None and x <= 5 for x in a):>8}   "
              + " ".join(f"{str(firsts[s] or '-'):>6}" for s in SHOW))


if __name__ == "__main__":
    main()
