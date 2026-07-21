from datetime import date, datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    source: str
    title: str
    url: str | None
    published_date: date | None
    ingested_at: datetime

    model_config = {"from_attributes": True}


class UploadDocumentMeta(BaseModel):
    source: str
    title: str
    url: str | None = None
    published_date: date | None = None
