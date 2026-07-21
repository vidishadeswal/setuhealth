from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.core.deps import require_admin
from backend.app.db.session import get_db
from backend.app.ingestion.pipeline import delete_document, ingest_document
from backend.app.models.document import Document
from backend.app.models.query_log import QueryLog, RefusalReason
from backend.app.models.user import User
from backend.app.schemas.documents import DocumentOut

router = APIRouter(prefix="/admin", tags=["admin"])

UPLOAD_DIR = Path(__file__).resolve().parents[3] / "corpus" / "uploads"


@router.post("/documents", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    source: str = Form(...),
    title: str = Form(...),
    url: str | None = Form(None),
    published_date: date | None = Form(None),
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> Document:
    # .name strips any directory components (including "../" traversal) — a bare
    # filename join here would let a crafted filename write outside UPLOAD_DIR, or
    # (via pathlib's absolute-path override behavior) anywhere on disk at all.
    safe_filename = Path(file.filename or "").name
    if not safe_filename or Path(safe_filename).suffix.lower() not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt sources are supported (plain-text extraction, no OCR)")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / safe_filename
    dest.write_bytes(await file.read())

    document = ingest_document(db, dest, source=source, title=title, url=url)
    if published_date:
        document.published_date = published_date
        db.commit()
    return document


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(document_id: str, admin: User = Depends(require_admin), db: Session = Depends(get_db)) -> None:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    delete_document(db, document)


@router.get("/eval-report")
def eval_report(admin: User = Depends(require_admin), db: Session = Depends(get_db)) -> dict:
    """Live operational metrics from queries_log. For the offline retrieval
    precision/recall, hallucination-rate, and red-team false-negative-rate numbers,
    run `python -m eval.run_eval` — those are computed against known-correct answers,
    which live query traffic doesn't have.
    """
    total = db.query(func.count(QueryLog.id)).scalar() or 0
    refused = db.query(func.count(QueryLog.id)).filter(QueryLog.was_refused.is_(True)).scalar() or 0
    avg_confidence = db.query(func.avg(QueryLog.confidence_score)).filter(QueryLog.was_refused.is_(False)).scalar()
    total_ungrounded = db.query(func.sum(QueryLog.ungrounded_claim_count)).scalar() or 0

    reason_breakdown = {
        reason.value: db.query(func.count(QueryLog.id)).filter(QueryLog.refusal_reason == reason).scalar() or 0
        for reason in RefusalReason
    }

    return {
        "total_queries": total,
        "answered_count": total - refused,
        "refused_count": refused,
        "refusal_rate": (refused / total) if total else None,
        "refusal_reason_breakdown": reason_breakdown,
        "avg_confidence_score_answered": float(avg_confidence) if avg_confidence is not None else None,
        "total_ungrounded_claims_flagged": int(total_ungrounded),
        "static_eval_command": "python -m eval.run_eval",
    }
