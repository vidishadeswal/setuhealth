"""In-process holder for the live FAISS + BM25 indexes. Rebuilt from the DB at app
startup and refreshed after any admin ingestion/deletion — at this corpus size (a
handful of documents, a few hundred chunks) a full refresh is milliseconds, so there's
no need for incremental index-update machinery.
"""

from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.core.query_cache import query_cache
from backend.app.models.chunk import Chunk
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.vector_index import VectorIndex


class RetrievalRegistry:
    def __init__(self):
        self.vector_index: VectorIndex | None = None
        self.bm25_index: BM25Index | None = None
        self.faiss_id_to_chunk_id: dict[int, str] = {}

    def refresh(self, db: Session) -> None:
        settings = get_settings()
        self.vector_index = VectorIndex.load(settings.index_dir / "vector.faiss")

        rows = db.query(Chunk.id, Chunk.faiss_id, Chunk.content).all()
        self.faiss_id_to_chunk_id = {r.faiss_id: r.id for r in rows if r.faiss_id is not None}

        self.bm25_index = BM25Index()
        self.bm25_index.rebuild([(r.id, r.content) for r in rows])

        # Any corpus change can invalidate a previously-cached answer (a source could
        # have just been deleted out from under it), and this is the one place every
        # ingestion/deletion path already funnels through.
        query_cache.clear()

    def get_retriever(self) -> HybridRetriever:
        if self.vector_index is None or self.bm25_index is None:
            raise RuntimeError("Retrieval indexes not loaded — call registry.refresh() first")
        return HybridRetriever(self.vector_index, self.bm25_index, self.faiss_id_to_chunk_id)


registry = RetrievalRegistry()
