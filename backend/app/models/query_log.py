import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class RefusalReason(str, enum.Enum):
    low_confidence = "low_confidence"
    emergency_flagged = "emergency_flagged"
    out_of_scope = "out_of_scope"


class QueryLog(Base):
    __tablename__ = "queries_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    was_refused: Mapped[bool] = mapped_column(Boolean, default=False)
    refusal_reason: Mapped[RefusalReason | None] = mapped_column(Enum(RefusalReason), nullable=True)
    ungrounded_claim_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
