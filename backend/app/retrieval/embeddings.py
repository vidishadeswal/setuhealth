"""Local embedding model (design doc Section 08): BAAI/bge-small-en-v1.5, chosen for its
retrieval-specific training and CPU-friendly size over a hosted API — no query or chunk
text ever leaves the machine.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.config import get_settings

# BGE models are trained with an asymmetric convention: passages are embedded as-is,
# but queries need this instruction prefix to get correct retrieval-quality vectors.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _get_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model_name)


def embed_passages(texts: list[str]) -> np.ndarray:
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype("float32")


def embed_query(text: str) -> np.ndarray:
    model = _get_model()
    vector = model.encode(BGE_QUERY_PREFIX + text, normalize_embeddings=True, convert_to_numpy=True)
    return vector.astype("float32")
