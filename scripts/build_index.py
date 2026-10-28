"""Build the per-contract indexes and run the retrieval smoke test.

    .venv/bin/python scripts/build_index.py [--no-build]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.chunker import section_matches  # noqa: E402
from legal_rag.indexer import ContractIndex, bm25_tokens, build_index  # noqa: E402

KS = (1, 3, 5)

# Hand-written queries that name a clause; BM25 must find them via the intact id token.
ID_QUERIES = [
    ("lic", "What does Section 7.4 say?", "7.4"),
    ("dis", "What does clause 15.B say?", "15.B"),
    ("svc", "What does Section 3.5 say?", "3.5"),
    ("sup", "What does Section 12.3 say?", "12.3"),
    ("jv", "What does clause 9 say?", "9"),
]


def hits(results, expected):
    """(any expected section in top-k, fraction of expected sections covered)."""
    cov = [any(section_matches(s, c["clause_ids"]) for c, _ in results) for s in expected]
    return any(cov), sum(cov) / len(cov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    meta = json.loads((ROOT / "data" / "contracts_meta.json").read_text())
    cid_by_file = {stem: m["contract_id"] for stem, m in meta.items()}

    if not args.no_build:
        for cid in cid_by_file.values():
            print("built", build_index(cid))

    indexes = {v: {c: ContractIndex(c, v) for c in cid_by_file.values()} for v in ("okapi", "plus")}
    for cid, ix in indexes["okapi"].items():
        n = len(ix.chunks)
        assert ix.vecs.shape[0] == n and all(c["contract_id"] == cid for c in ix.chunks)
        print(f"{cid}: {n} chunks, dim={ix.vecs.shape[1]}, norm range "
              f"{float((ix.vecs ** 2).sum(1).min()):.3f}-{float((ix.vecs ** 2).sum(1).max()):.3f}")

    rows = [json.loads(l) for l in open(ROOT / "golden_set.jsonl")]
    rows = [r for r in rows if r["expected_sections"]]
    modes = {
        "dense + prefix": lambda ix, q, k: ix.dense(q, k, use_prefix=True),
        "dense no prefix": lambda ix, q, k: ix.dense(q, k, use_prefix=False),
        "bm25 okapi": lambda ix, q, k: ix.sparse(q, k),
        "bm25 plus": lambda ix, q, k: ix.sparse(q, k),
    }
    print(f"\n== golden retrieval ({len(rows)} questions with expected sections) ==")
    print(f"{'mode':16}" + "".join(f"hit@{k:<3} " for k in KS) + "".join(f"rec@{k:<3} " for k in KS))
    misses = {}
    for name, fn in modes.items():
        variant = "plus" if name == "bm25 plus" else "okapi"
        h = {k: 0 for k in KS}
        r = {k: 0.0 for k in KS}
        misses[name] = []
        for g in rows:
            ix = indexes[variant][cid_by_file[g["cuad_file"]]]
            res = fn(ix, g["question"], max(KS))
            for k in KS:
                ok, rec = hits(res[:k], g["expected_sections"])
                h[k] += ok
                r[k] += rec
            if not hits(res[:5], g["expected_sections"])[0]:
                misses[name].append(g["id"])
        n = len(rows)
        print(f"{name:16}" + "".join(f"{h[k] / n:<7.2f}" for k in KS) + "".join(f"{r[k] / n:<7.2f}" for k in KS))
    for name, m in misses.items():
        print(f"  misses@5 {name}: {m}")

    print("\n== clause-id queries (BM25 okapi / plus / dense top-1) ==")
    for cid, q, want in ID_QUERIES:
        out = []
        for label, res in (("okapi", indexes["okapi"][cid].sparse(q, 1)),
                           ("plus", indexes["plus"][cid].sparse(q, 1)),
                           ("dense", indexes["okapi"][cid].dense(q, 1))):
            c = res[0][0]
            out.append(f"{label}={c['clause_id']}{'✓' if section_matches(want, c['clause_ids']) else '✗'}")
        print(f"{cid:4} {q:32} want {want:5} {'  '.join(out)}")

    print("\nsample tokens:", bm25_tokens("Section 7.4(a) and 15.B apply; see clause 12.3, 3(b) and Ohio law, 2015."))


if __name__ == "__main__":
    main()
