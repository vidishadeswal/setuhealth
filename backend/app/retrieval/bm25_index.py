"""BM25 keyword index (design doc Section 09) — catches exact drug names and dosages that
dense embeddings under-weight. rank_bm25 has no incremental add/remove API, so the index
is rebuilt from the current chunk set on ingestion/deletion; at this corpus size (a
handful of documents) that's milliseconds, not a real cost.
"""

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    def rebuild(self, chunks: list[tuple[str, str]]) -> None:
        """chunks: list of (chunk_id, content)."""
        self._chunk_ids = [c[0] for c in chunks]
        tokenized = [tokenize(c[1]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:k] if s > 0]
