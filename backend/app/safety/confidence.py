"""Confidence scoring (design doc Section 12): computed from retrieval signals before
generation ever runs. Combines the top reranked chunk's relevance, the score margin
between the top-1 and top-2 chunks (a small margin means retrieval itself is unsure),
and how much BM25 and vector search agreed. The threshold this is compared against is
calibrated empirically — see eval/run_eval.py's threshold sweep — not chosen by feel.
"""

import math
from dataclasses import dataclass

from backend.app.retrieval.hybrid import FusedCandidate
from backend.app.retrieval.reranker import RerankedChunk

WEIGHT_TOP_SCORE = 0.6
WEIGHT_MARGIN = 0.25
WEIGHT_AGREEMENT = 0.15


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class ConfidenceResult:
    score: float
    top_score_component: float
    margin_component: float
    agreement_component: float


def compute_confidence(
    reranked: list[RerankedChunk],
    fused_candidates: list[FusedCandidate],
) -> ConfidenceResult:
    if not reranked:
        return ConfidenceResult(0.0, 0.0, 0.0, 0.0)

    top_score = _sigmoid(reranked[0].rerank_score)
    second_score = _sigmoid(reranked[1].rerank_score) if len(reranked) > 1 else 0.0
    margin = top_score - second_score if len(reranked) > 1 else top_score

    agreement_by_id = {c.chunk_id: (c.in_bm25 and c.in_vector) for c in fused_candidates}
    top_n = reranked[: min(3, len(reranked))]
    agreement = sum(1 for r in top_n if agreement_by_id.get(r.chunk_id, False)) / len(top_n)

    score = WEIGHT_TOP_SCORE * top_score + WEIGHT_MARGIN * margin + WEIGHT_AGREEMENT * agreement
    return ConfidenceResult(
        score=max(0.0, min(1.0, score)),
        top_score_component=top_score,
        margin_component=margin,
        agreement_component=agreement,
    )
