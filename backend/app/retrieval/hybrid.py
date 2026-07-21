"""Hybrid retrieval (design doc Section 09): BM25 and vector search each produce a ranked
list; reciprocal rank fusion combines them so a chunk both methods independently surface
gets rewarded, rather than just concatenating result sets.
"""

from dataclasses import dataclass

from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.embeddings import embed_query
from backend.app.retrieval.vector_index import VectorIndex

RRF_K = 60  # standard constant; de-weights rank position without a corpus-specific tune


@dataclass
class FusedCandidate:
    chunk_id: str
    fused_score: float
    in_bm25: bool
    in_vector: bool


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Ties the BM25 index and FAISS index together. faiss_id_to_chunk_id resolves the
    integer ids the vector index speaks into the chunk primary keys BM25 and the DB use.
    """

    def __init__(self, vector_index: VectorIndex, bm25_index: BM25Index, faiss_id_to_chunk_id: dict[int, str]):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.faiss_id_to_chunk_id = faiss_id_to_chunk_id

    def retrieve(self, query: str, top_k_per_method: int, top_k_fused: int) -> list[FusedCandidate]:
        bm25_hits = self.bm25_index.search(query, top_k_per_method)
        bm25_ranked_ids = [cid for cid, _ in bm25_hits]

        query_vector = embed_query(query)
        vector_hits = self.vector_index.search(query_vector, top_k_per_method)
        vector_ranked_ids = [self.faiss_id_to_chunk_id[fid] for fid, _ in vector_hits if fid in self.faiss_id_to_chunk_id]

        fused_scores = reciprocal_rank_fusion([bm25_ranked_ids, vector_ranked_ids])

        bm25_set, vector_set = set(bm25_ranked_ids), set(vector_ranked_ids)
        candidates = [
            FusedCandidate(
                chunk_id=cid,
                fused_score=score,
                in_bm25=cid in bm25_set,
                in_vector=cid in vector_set,
            )
            for cid, score in fused_scores.items()
        ]
        candidates.sort(key=lambda c: c.fused_score, reverse=True)
        return candidates[:top_k_fused]
