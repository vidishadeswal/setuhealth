"""Offline ingestion pipeline (design doc Section 04): PDFs/text -> extraction -> chunking
-> embedding -> FAISS + BM25 indexing -> Postgres/SQLite rows. Runs synchronously; at
5-10 source documents this is a matter of seconds, not a background-job problem.
"""

from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.ingestion.chunker import chunk_page
from backend.app.ingestion.extract import extract_pages
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.retrieval.embeddings import embed_passages
from backend.app.retrieval.registry import registry
from backend.app.retrieval.vector_index import VectorIndex


def _grounded_chunk_content(content: str, page_number: int, drug_name: str) -> str:
    """FDA label prose often names the drug once at the top of a section and refers to
    it only by drug class afterward ("...should not be given with diuretics...") —
    correct in context, but once that sentence is split into its own chunk (see
    build_corpus_from_fda.py), the connection to the specific drug is gone. A real
    case: furosemide's Lithium interaction never says "furosemide" at all, so a
    cross-encoder scoring it against "does furosemide interact with lithium" ranked
    it dead last behind four completely unrelated furosemide passages that happened
    to repeat the drug's name. Prepending the drug name to every chunk's stored
    content (not just page 1, the attribution header, which doesn't need it) keeps
    that grounding regardless of how the source prose happens to be worded — this
    affects retrieval, reranking, and what the LLM prompt sees, since all three read
    chunk.content directly.
    """
    if page_number == 1:
        return content
    return f"{drug_name}: {content}"


def ingest_document(db: Session, path: Path, source: str, title: str, url: str | None = None) -> Document:
    settings = get_settings()

    document = Document(source=source, title=title, url=url)
    db.add(document)
    db.flush()  # assigns document.id without committing

    drug_name = title.removesuffix(" Interactions")

    pages = extract_pages(path)
    all_chunks: list[Chunk] = []
    all_texts: list[str] = []

    for page_number, page_text in pages:
        for result in chunk_page(page_text, page_number, settings.chunk_token_size, settings.chunk_overlap_ratio):
            content = _grounded_chunk_content(result.content, result.page_number, drug_name)
            chunk = Chunk(doc_id=document.id, content=content, page_number=result.page_number)
            all_chunks.append(chunk)
            all_texts.append(content)

    if all_chunks:
        vectors = embed_passages(all_texts)
        next_faiss_id = _next_available_faiss_id(db)
        for i, chunk in enumerate(all_chunks):
            chunk.faiss_id = next_faiss_id + i
        db.add_all(all_chunks)
        db.flush()

        vector_index = load_vector_index(settings.index_dir)
        vector_index.add(vectors, [c.faiss_id for c in all_chunks])
        vector_index.save(settings.index_dir / "vector.faiss")

    db.commit()
    registry.refresh(db)
    return document


def delete_document(db: Session, document: Document) -> None:
    settings = get_settings()
    faiss_ids = [c.faiss_id for c in document.chunks if c.faiss_id is not None]

    vector_index = load_vector_index(settings.index_dir)
    vector_index.remove(faiss_ids)
    vector_index.save(settings.index_dir / "vector.faiss")

    db.delete(document)
    db.commit()
    registry.refresh(db)


def load_vector_index(index_dir: Path) -> VectorIndex:
    return VectorIndex.load(index_dir / "vector.faiss")


def _next_available_faiss_id(db: Session) -> int:
    # MAX() ignores NULLs uniformly across backends. An ORDER BY ... DESC LIMIT 1
    # doesn't: Postgres sorts NULLs first on DESC by default while SQLite sorts
    # them last, so on Postgres this would silently return the wrong (NULL) row
    # whenever any chunk lacks a faiss_id, and hand out an already-used id.
    max_id = db.query(func.max(Chunk.faiss_id)).scalar()
    return (max_id + 1) if max_id is not None else 0
