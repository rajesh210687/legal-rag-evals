"""Search entry points. Both are scoped to one contract."""
from functools import lru_cache

from .indexer import ContractIndex, names_clause

DEFAULT_TOP_K = 8  # excerpts passed to the prompt; 8 covered multi-clause questions that 5 missed (evals/runs)


@lru_cache(maxsize=None)
def _index(contract_id: str) -> ContractIndex:
    return ContractIndex(contract_id, bm25_variant="plus")


def search_embeddings(contract_id: str, q: str, k: int = 20):
    """Dense search with the bge query prefix. Returns [(chunk, score)]."""
    return _index(contract_id).dense(q, k, use_prefix=True)


def search_hybrid(contract_id: str, q: str, k: int = 20, rrf_k: int = 10, dense_weight: float | None = None):
    """Weighted reciprocal rank fusion: score = sum(weight / (rrf_k + rank)). Returns [(chunk, score)].

    dense_weight defaults by query type: a query that names a clause leans on BM25 (0.25),
    anything else leans on the embeddings (2.0). Both choices come from scripts/rrf_sweep.py.
    """
    if dense_weight is None:
        dense_weight = 0.25 if names_clause(q) else 2.0
    depth = len(_index(contract_id).chunks)  # rank every chunk, so a method's weak hit still counts
    fused, by_id = {}, {}
    runs = ((search_embeddings(contract_id, q, depth), dense_weight), (search_bm25(contract_id, q, depth), 1.0))
    for results, weight in runs:
        for rank, (chunk, _) in enumerate(results, 1):
            by_id[chunk["chunk_id"]] = chunk
            fused[chunk["chunk_id"]] = fused.get(chunk["chunk_id"], 0.0) + weight / (rrf_k + rank)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(by_id[i], fused[i]) for i in top]


def search_bm25(contract_id: str, q: str, k: int = 20):
    """BM25Plus over lowercase tokens with clause ids kept whole. Returns [(chunk, score)]."""
    return _index(contract_id).sparse(q, k)
