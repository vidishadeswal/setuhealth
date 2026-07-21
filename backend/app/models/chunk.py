import uuid

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.base import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # The embedding vector itself lives in the FAISS index (retrieval/vector_index.py),
    # not in this row — faiss_id is the join key between the two.
    faiss_id: Mapped[int | None] = mapped_column(Integer, unique=True, nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")  # noqa: F821
