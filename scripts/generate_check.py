"""End-to-end check: retrieve the top chunks, generate an answer, run the citation checks.

    .venv/bin/python scripts/generate_check.py [--ids LIC-01 SVC-05 LIC-06]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.chunker import section_matches  # noqa: E402
from legal_rag.generate import excerpt_id, generate, get_client, save_run  # noqa: E402
from legal_rag.retrieval import DEFAULT_TOP_K, search_hybrid  # noqa: E402


def top_chunks(contract_id, question):
    return [c for c, _ in search_hybrid(contract_id, question, DEFAULT_TOP_K)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=["LIC-01", "SVC-05", "LIC-06"])
    ap.add_argument("--no-fallbacks", action="store_true")
    ap.add_argument("--out", default="evals/runs/generate_check.jsonl")
    args = ap.parse_args()

    meta = json.loads((ROOT / "data" / "contracts_meta.json").read_text())
    cid = {k: v["contract_id"] for k, v in meta.items()}
    gold = {g["id"]: g for g in (json.loads(l) for l in open(ROOT / "golden_set.jsonl"))}
    client = get_client()

    for gid in args.ids:
        g = gold[gid]
        chunks = top_chunks(cid[g["cuad_file"]], g["question"])
        res = generate(g["question"], chunks, client=client, use_fallbacks=not args.no_fallbacks)
        res.update(id=gid, expected_sections=g["expected_sections"], expected_answer=g["expected_answer"])
        by_id = {excerpt_id(c): c for c in chunks}
        cited = (res["answer"] or {}).get("citations", [])
        res["expected_sections_retrieved"] = [
            s for s in g["expected_sections"] if any(section_matches(s, c["clause_ids"]) for c in chunks)
        ]
        res["cited_cover_expected"] = [
            s for s in g["expected_sections"]
            if any(section_matches(s, by_id[c]["clause_ids"]) for c in cited if c in by_id)
        ]
        save_run(res, ROOT / args.out)

        print(f"\n{'=' * 78}\n{gid} [{g['type']}] {g['question']}")
        print(f"sent excerpts: {res['excerpt_ids']}")
        print(f"golden:  {g['expected_answer']}\n         sections={g['expected_sections']}")
        if res["answer"]:
            a = res["answer"]
            print(f"model:   {a['answer']}\n         citations={a['citations']} found={a['found']}")
        print(f"error={res['error']} first_error={res['first_error']} attempts={res['attempts']} "
              f"stop={res['stop_reason']} model={res['model']}")
        print(f"checks={res['checks']}")
        print(f"expected sections in the excerpts sent: {res['expected_sections_retrieved']} | "
              f"covered by citations: {res['cited_cover_expected']}")
        print(f"tokens in/out={res['usage']['input_tokens']}/{res['usage']['output_tokens']}  {res['seconds']:.1f}s")


if __name__ == "__main__":
    main()
