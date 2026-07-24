"""Cross-encoder reranking (design doc Section 10): scores query+chunk together instead
of comparing independently-encoded vectors, catching fine-grained relevance the bi-encoder
retrieval step misses. Run only over the fused top-K candidates, not the whole corpus.
"""

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from backend.app.config import get_settings


@lru_cache
def _get_reranker() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.reranker_model_name)


@dataclass
class RerankedChunk:
    chunk_id: str
    content: str
    page_number: int | None
    document_title: str
    rerank_score: float


def rerank(query: str, candidates: list[dict], top_k: int) -> list[RerankedChunk]:
    """candidates: list of dicts with chunk_id, content, page_number, document_title."""
    return rerank_multi([query], candidates, top_k)


def rerank_multi(queries: list[str], candidates: list[dict], top_k: int) -> list[RerankedChunk]:
    """Scores every candidate against every query variant and keeps each candidate's
    best score across them. Used by the LLM query-rewrite fallback (api/routes_ask.py):
    widening *retrieval* with paraphrases is wasted if reranking still only judges
    relevance against the one original phrasing the cross-encoder happened to score
    low — a paraphrase only needs to match the passage well under ONE of its phrasings
    to prove the chunk is actually relevant. A single query is the N=1 case, which is
    why rerank() is a one-line wrapper around this rather than separate logic.
    """
    if not candidates:
        return []

    model = _get_reranker()
    best_scores = [float("-inf")] * len(candidates)
    for query in queries:
        pairs = [(query, c["content"]) for c in candidates]
        scores = model.predict(pairs)
        for i, score in enumerate(scores):
            if score > best_scores[i]:
                best_scores[i] = float(score)

    scored = [
        RerankedChunk(
            chunk_id=c["chunk_id"],
            content=c["content"],
            page_number=c.get("page_number"),
            document_title=c["document_title"],
            rerank_score=best_scores[i],
        )
        for i, c in enumerate(candidates)
    ]
    scored.sort(key=lambda r: r.rerank_score, reverse=True)
    return scored[:top_k]
