from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str


class Citation(BaseModel):
    document_title: str
    page_number: int | None
    chunk_id: str
    content: str


class EmergencyResource(BaseModel):
    label: str
    value: str


class AskResponse(BaseModel):
    status: str  # "answered" | "refused" | "emergency"
    answer: str | None = None
    citations: list[Citation] = []
    confidence_score: float | None = None
    refusal_reason: str | None = None
    emergency_resources: list[EmergencyResource] = []
