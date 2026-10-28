"""Run all golden questions through retrieval + generation and log one row per question.

    .venv/bin/python scripts/run_eval.py --tag my-run [--k 8] [--fresh]

Writes evals/runs/<tag>/results.jsonl, run_config.json and summary.json.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from legal_rag import generate as gen  # noqa: E402
from legal_rag.evaluation import clause_ranks, covered, first_and_worst  # noqa: E402
from legal_rag.indexer import ContractIndex  # noqa: E402
from legal_rag.retrieval import DEFAULT_TOP_K, search_bm25, search_embeddings, search_hybrid  # noqa: E402

DEPTH = 1000  # rank every chunk so a late hit still gets a number
PRICE_IN, PRICE_OUT = 2.00, 10.00  # $/1M tokens, Sonnet 5.5


def git_state():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
        return {"commit": sha, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def label(row, answerable):
    """Why a question did or didn't pass."""
    if row["error"] or row["answer"] is None:
        return "generation_error"
    if not answerable:
        return "ok" if row["found_false_correct"] else "false_positive"
    if not row["golden_in_prompt_any"]:
        return "retrieval_miss"
    if row["answer"]["found"] is False:
        return "said_not_found_despite_golden"
    if not row["checks"]["ok"]:
        return "bad_citations"
    if not row["cites_golden_any"]:
        return "generation_miss"
    if not (row["golden_in_prompt_all"] and row["cites_golden_all"]):
        return "partial"
    return "ok"


def run_question(g, cid, k, client):
    q = g["question"]
    expected = g["expected_sections"]
    answerable = bool(expected)
    t0 = time.time()
    bm25 = search_bm25(cid, q, DEPTH)
    emb = search_embeddings(cid, q, DEPTH)
    fused = search_hybrid(cid, q, DEPTH)
    t_retrieval = time.time() - t0
    chunks = [c for c, _ in fused[:k]]

    row = {"id": g["id"], "type": g["type"], "question": q, "expected_sections": expected,
           "expected_answer": g["expected_answer"], "k": k, "excerpt_ids": [gen.excerpt_id(c) for c in chunks]}
    if answerable:
        ranks = {"bm25": clause_ranks(bm25, expected), "embeddings": clause_ranks(emb, expected),
                 "fused": clause_ranks(fused, expected)}
        row["ranks"] = ranks
        row["rank_first_worst"] = {m: first_and_worst(r) for m, r in ranks.items()}
        in_prompt = covered(chunks, expected)
        row["golden_in_prompt_any"] = bool(in_prompt)
        row["golden_in_prompt_all"] = len(in_prompt) == len(expected)

    res = gen.generate(q, chunks, client=client)
    by_id = {gen.excerpt_id(c): c for c in chunks}
    ans = res["answer"]
    cited_chunks = [by_id[c] for c in (ans["citations"] if ans else []) if c in by_id]
    row.update(answer=ans, checks=res["checks"], error=res["error"], first_error=res["first_error"],
               attempts=res["attempts"], model=res["model"], stop_reason=res["stop_reason"],
               input_tokens=res["usage"]["input_tokens"], output_tokens=res["usage"]["output_tokens"],
               seconds_model=round(res["seconds"], 2), seconds_retrieval=round(t_retrieval, 2))
    if answerable:
        cov = covered(cited_chunks, expected)
        row["cited_golden"] = cov
        row["cites_golden_any"] = bool(cov)
        row["cites_golden_all"] = len(cov) == len(expected)
    else:
        row["found_false_correct"] = bool(ans) and ans["found"] is False
    row["label"] = label(row, answerable)
    return row


def summarize(rows):
    ans = [r for r in rows if r["expected_sections"]]
    neg = [r for r in rows if not r["expected_sections"]]
    pct = lambda xs: (100.0 * sum(xs) / len(xs)) if xs else float("nan")
    retrieved = [r for r in ans if r["golden_in_prompt_any"]]
    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)
    labels = {}
    for r in rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    return {
        "n_answerable": len(ans), "n_unanswerable": len(neg),
        "retrieval_recall_at_k_any": pct([r["golden_in_prompt_any"] for r in ans]),
        "retrieval_recall_at_k_all": pct([r["golden_in_prompt_all"] for r in ans]),
        "citation_accuracy_any": pct([r["cites_golden_any"] for r in ans]),
        "citation_accuracy_all": pct([r["cites_golden_all"] for r in ans]),
        "citation_accuracy_any_given_retrieved": pct([r["cites_golden_any"] for r in retrieved]),
        "unanswerable_found_false": pct([r["found_false_correct"] for r in neg]),
        "labels": labels,
        "citation_check_failures": sum(1 for r in rows if r["checks"] and not r["checks"]["ok"]),
        "errors": sum(1 for r in rows if r["error"]),
        "retries": sum(1 for r in rows if r["attempts"] > 1),
        "mean_seconds_model": sum(r["seconds_model"] for r in rows) / len(rows),
        "mean_seconds_retrieval": sum(r["seconds_retrieval"] for r in rows) / len(rows),
        "input_tokens": tin, "output_tokens": tout,
        "est_cost_usd": round(tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="excerpts passed to the prompt")
    ap.add_argument("--fresh", action="store_true", help="ignore rows already logged for this tag")
    args = ap.parse_args()

    out = ROOT / "evals" / "runs" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    if args.fresh and results_path.exists():
        results_path.unlink()
    done = {}
    if results_path.exists():
        done = {r["id"]: r for r in (json.loads(l) for l in results_path.read_text().splitlines() if l.strip())}

    meta = json.loads((ROOT / "data" / "contracts_meta.json").read_text())
    cid = {k: v["contract_id"] for k, v in meta.items()}
    golden = [json.loads(l) for l in open(ROOT / "golden_set.jsonl")]
    chunk_hash = {c: hashlib.sha256((ROOT / "data" / "chunks" / f"{c}.jsonl").read_bytes()).hexdigest()[:12]
                  for c in cid.values()}
    (out / "run_config.json").write_text(json.dumps({
        "tag": args.tag, "k": args.k, "started": datetime.now(timezone.utc).isoformat(),
        "model": gen.MODEL, "effort": gen.EFFORT, "max_tokens": gen.MAX_TOKENS, "fallbacks": True,
        "system_prompt_sha": hashlib.sha256(gen.SYSTEM_PROMPT.encode()).hexdigest()[:12],
        "retrieval": {"fusion": "weighted RRF, k=10; dense x2, or dense x0.25 when the query names a clause",
                      "bm25": "BM25Plus", "embedding_model": "BAAI/bge-base-en-v1.5"},
        "chunk_files_sha": chunk_hash, "git": git_state(),
    }, indent=2))

    client = gen.get_client()
    search_embeddings(next(iter(cid.values())), "warm up", 1)  # load the embedding model before timing anything

    rows = []
    for g in golden:
        if g["id"] in done:
            rows.append(done[g["id"]])
            continue
        row = run_question(g, cid[g["cuad_file"]], args.k, client)
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        print(f"{g['id']:7}{row['label']:30}found={(row['answer'] or {}).get('found')} "
              f"cites={(row['answer'] or {}).get('citations')}  {row['seconds_model']}s", flush=True)

    s = summarize(rows)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(f"\n== {args.tag}: k={args.k} ==")
    print(f"Retrieval recall@{args.k}:   any {s['retrieval_recall_at_k_any']:.0f}%   all {s['retrieval_recall_at_k_all']:.0f}%   (n={s['n_answerable']})")
    print(f"Citation accuracy:      any {s['citation_accuracy_any']:.0f}%   all {s['citation_accuracy_all']:.0f}%   "
          f"| given golden reached the prompt: {s['citation_accuracy_any_given_retrieved']:.0f}%")
    print(f"Unanswerable handling:  {s['unanswerable_found_false']:.0f}% marked found:false   (n={s['n_unanswerable']})")
    print(f"labels={s['labels']}  citation-check failures={s['citation_check_failures']}  errors={s['errors']}  retries={s['retries']}")
    print(f"model {s['mean_seconds_model']:.1f}s/q  retrieval {s['mean_seconds_retrieval']:.2f}s/q  "
          f"tokens in/out {s['input_tokens']}/{s['output_tokens']}  est. ${s['est_cost_usd']}")


if __name__ == "__main__":
    main()
