"""FAISS vector index (design doc Section 09). Embeddings are L2-normalized at encode
time, so inner product == cosine similarity — IndexFlatIP is used directly rather than
paying for an explicit cosine step. Wrapped in IndexIDMap2 so chunk.faiss_id values can
be used as stable ids across add/remove/reconstruct, which a bare IndexFlat doesn't support.
"""

from pathlib import Path

import faiss
import numpy as np

EMBEDDING_DIM = 384  # bge-small-en-v1.5 output dimension


class VectorIndex:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        self.index.add_with_ids(vectors, np.array(ids, dtype="int64"))

    def remove(self, ids: list[int]) -> None:
        if ids:
            self.index.remove_ids(np.array(ids, dtype="int64"))

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vector.reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path, dim: int = EMBEDDING_DIM) -> "VectorIndex":
        instance = cls(dim=dim)
        if path.exists():
            instance.index = faiss.read_index(str(path))
        return instance
