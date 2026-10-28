"""Per-contract dense (bge) and sparse (BM25) indexes over the clause chunks.

Retrieval is always scoped to one contract: ContractIndex is loaded for a single
contract_id, so there is no way to search across contracts.
"""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus

ROOT = Path(__file__).resolve().parent.parent.parent
CHUNK_DIR = ROOT / "data" / "chunks"
INDEX_DIR = ROOT / "data" / "index"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BM25_VARIANTS = {"okapi": BM25Okapi, "plus": BM25Plus}

# Clause numbers stay whole: 12.3, 7.4(a), 15.b, 3(b). Everything else is a plain word.
_TOKEN = re.compile(
    r"(?P<id>(?<![\w.])\d{1,2}(?:\.\d{1,3})+(?:\([a-z]\))*(?![\w])"
    r"|(?<![\w.])\d{1,2}\.[a-z](?![\w])(?:\([a-z]\))*"
    r"|(?<![\w.])\d{1,2}\([a-z]\))"
    r"|(?P<word>[a-z0-9]+(?:['’\-][a-z0-9]+)*)"
)


def bm25_tokens(text: str):
    """Lowercase tokens; a clause id also emits its parent (7.4(a) -> 7.4(a), 7.4; 15.b -> 15.b, 15)."""
    out = []
    for m in _TOKEN.finditer(text.lower()):
        if m.group("id"):
            tok = m.group("id")
            out.append(tok)
            if "(" in tok:
                out.append(tok.split("(", 1)[0])  # 7.4(a) -> 7.4, 3(b) -> 3
            elif re.fullmatch(r"\d+\.[a-z]", tok):
                out.append(tok.split(".", 1)[0])  # 15.b -> 15
        else:
            out.append(m.group("word"))
    return out


_CLAUSE_WORD = re.compile(r"(?:section|clause|article|paragraph|§)\s*\d{1,2}\b", re.I)


def names_clause(query: str) -> bool:
    """True if the query points at a clause: 12.3, 7.4(a), 15.B, "clause 9", "Section 16"."""
    return bool(_CLAUSE_WORD.search(query)) or any(m.group("id") for m in _TOKEN.finditer(query.lower()))


def bm25_doc(chunk: dict) -> str:
    """Section ids + heading + body. The title/parties header is left out: it repeats in every chunk."""
    ids = " ".join(chunk["clause_ids"])
    return f"{ids} {chunk['section_heading']} {chunk['body']}"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(MODEL_NAME)
    m.max_seq_length = 512
    return m


def _chunks_path(contract_id: str) -> Path:
    return CHUNK_DIR / f"{contract_id}.jsonl"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_chunks(contract_id: str):
    return [json.loads(l) for l in _chunks_path(contract_id).read_text().splitlines() if l.strip()]


def build_index(contract_id: str):
    path = _chunks_path(contract_id)
    chunks = load_chunks(contract_id)
    vecs = _model().encode(
        [c["text"] for c in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)  # chunks get no query prefix
    out = INDEX_DIR / contract_id
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vecs)
    (out / "tokens.json").write_text(json.dumps([bm25_tokens(bm25_doc(c)) for c in chunks]))
    (out / "manifest.json").write_text(json.dumps({
        "contract_id": contract_id, "n_chunks": len(chunks), "dim": int(vecs.shape[1]),
        "model": MODEL_NAME, "query_prefix": QUERY_PREFIX, "chunks_sha256": _sha(path),
    }, indent=2))
    return out


class ContractIndex:
    """Dense and BM25 search over a single contract."""

    def __init__(self, contract_id: str, bm25_variant: str = "okapi"):
        d = INDEX_DIR / contract_id
        manifest = json.loads((d / "manifest.json").read_text())
        if manifest["chunks_sha256"] != _sha(_chunks_path(contract_id)):
            raise RuntimeError(f"index for {contract_id} is stale; rebuild it")
        self.contract_id = contract_id
        self.chunks = load_chunks(contract_id)
        self.vecs = np.load(d / "embeddings.npy")
        assert len(self.chunks) == len(self.vecs)
        self.bm25 = BM25_VARIANTS[bm25_variant](json.loads((d / "tokens.json").read_text()))

    def _top(self, scores, k):
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def dense(self, query: str, k: int = 5, use_prefix: bool = True):
        q = _model().encode(
            [(QUERY_PREFIX + query) if use_prefix else query], normalize_embeddings=True
        ).astype(np.float32)[0]
        return self._top(self.vecs @ q, k)

    def sparse(self, query: str, k: int = 5):
        return self._top(np.asarray(self.bm25.get_scores(bm25_tokens(query))), k)
