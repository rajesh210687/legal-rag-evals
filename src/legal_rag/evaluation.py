"""Shared helpers for scoring retrieval and citations against the golden set."""
from .chunker import section_matches


def clause_ranks(results, expected):
    """results: [(chunk, score)] best first. Returns {golden clause: rank of its first chunk, or None}."""
    return {
        s: next((i for i, (c, _) in enumerate(results, 1) if section_matches(s, c["clause_ids"])), None)
        for s in expected
    }


def first_and_worst(ranks: dict):
    """(rank of the first golden clause found, rank at which every golden clause is found)."""
    found = [r for r in ranks.values() if r]
    first = min(found) if found else None
    worst = max(found) if found and len(found) == len(ranks) else None
    return first, worst


def covered(chunks, expected):
    """Golden clauses that appear in at least one of the given chunks."""
    return [s for s in expected if any(section_matches(s, c["clause_ids"]) for c in chunks)]
