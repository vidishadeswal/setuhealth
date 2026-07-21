from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.deps import require_agent
from backend.app.db.session import get_db
from backend.app.models.document import Document
from backend.app.models.user import User
from backend.app.schemas.documents import DocumentOut

router = APIRouter(tags=["sources"])


@router.get("/sources", response_model=list[DocumentOut])
def list_sources(user: User = Depends(require_agent), db: Session = Depends(get_db)) -> list[Document]:
    return db.query(Document).order_by(Document.ingested_at.desc()).all()
