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
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    scored = [
        RerankedChunk(
            chunk_id=c["chunk_id"],
            content=c["content"],
            page_number=c.get("page_number"),
            document_title=c["document_title"],
            rerank_score=float(score),
        )
        for c, score in zip(candidates, scores)
    ]
    scored.sort(key=lambda r: r.rerank_score, reverse=True)
    return scored[:top_k]
