"""Search entry points. Both are scoped to one contract."""
from functools import lru_cache

from .indexer import ContractIndex


@lru_cache(maxsize=None)
def _index(contract_id: str) -> ContractIndex:
    return ContractIndex(contract_id, bm25_variant="plus")


def search_embeddings(contract_id: str, q: str, k: int = 20):
    """Dense search with the bge query prefix. Returns [(chunk, score)]."""
    return _index(contract_id).dense(q, k, use_prefix=True)


def search_bm25(contract_id: str, q: str, k: int = 20):
    """BM25Plus over lowercase tokens with clause ids kept whole. Returns [(chunk, score)]."""
    return _index(contract_id).sparse(q, k)
