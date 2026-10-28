"""Chunk every contract, write data/chunks/<contract_id>.jsonl and print the checks.

    .venv/bin/python scripts/build_chunks.py [--sample 10]
"""
import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.chunker import MAX_TOKENS, chunk_contract, section_matches  # noqa: E402


def norm(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", s).lower().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="print N random chunks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((ROOT / "data" / "contracts_meta.json").read_text())
    out_dir = ROOT / "data" / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_file, all_chunks = {}, []
    for stem, m in meta.items():
        raw = (ROOT / "contracts" / f"{stem}.txt").read_text(errors="replace")
        chunks = chunk_contract(raw, m["contract_id"], stem, m["title"], m["parties"])
        by_file[stem] = chunks
        all_chunks += chunks
        with open(out_dir / f"{m['contract_id']}.jsonl", "w") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict()) + "\n")

    print("== chunks and token lengths (bge, incl. header) ==")
    for stem, chunks in by_file.items():
        t = sorted(c.token_count for c in chunks)
        top = len({c.clause_id.split(".")[0] for c in chunks if not c.clause_id.startswith(("pre", "back"))})
        print(f"{meta[stem]['contract_id']:4} chunks={len(chunks):4} top-level clauses={top:3} "
              f"min={t[0]:3} median={int(statistics.median(t)):3} p95={t[int(len(t) * .95) - 1]:3} max={t[-1]:3} "
              f">{MAX_TOKENS}={sum(x > MAX_TOKENS for x in t)}")

    print("\n== golden set coverage ==")
    bad = 0
    for line in open(ROOT / "golden_set.jsonl"):
        g = json.loads(line)
        chunks = by_file[g["cuad_file"]]
        secs = g["expected_sections"]
        if not secs:
            continue
        missing = [s for s in secs if not any(section_matches(s, c.clause_ids) for c in chunks)]
        scope = [c for c in chunks if any(section_matches(s, c.clause_ids) for s in secs)]
        parts = [p for p in re.split(r"\s*(?:\.\.\.|…)\s*", g["source_span"]) if p.strip()]
        lost = [p[:60] for p in parts if not any(norm(p).strip(" .") in norm(c.body) for c in scope)]
        if missing or lost:
            bad += 1
            print(f"{g['id']}: missing sections={missing} span not in matching chunks={lost}")
    print("all golden sections/spans found" if not bad else f"{bad} golden rows with problems")

    if args.sample:
        print("\n== random chunks ==")
        for c in random.Random(args.seed).sample(all_chunks, args.sample):
            print(f"--- {c.chunk_id} ids={c.clause_ids} tokens={c.token_count}\n{c.text[:600]}\n")


if __name__ == "__main__":
    main()
