from dataclasses import dataclass, field

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.config import Settings, get_settings
from backend.app.core.deps import require_agent
from backend.app.core.query_cache import query_cache
from backend.app.core.rate_limit import enforce_ask_rate_limit
from backend.app.db.session import get_db
from backend.app.generation.llm_client import generate
from backend.app.generation.prompt import build_prompt
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.models.query_log import QueryLog, RefusalReason
from backend.app.models.user import User
from backend.app.retrieval.query_expansion import expand_query
from backend.app.retrieval.registry import registry
from backend.app.retrieval.reranker import rerank
from backend.app.safety.confidence import compute_confidence
from backend.app.safety.emergency import check_emergency
from backend.app.safety.groundedness import check_groundedness
from backend.app.schemas.ask import AskRequest, AskResponse, Citation, EmergencyResource

router = APIRouter(tags=["ask"])

EMERGENCY_RESOURCES = [
    EmergencyResource(label="Emergency", value="Call your local emergency number now (e.g. 108 in India, 911 in the US, 999 in the UK)."),
    EmergencyResource(label="Nearest hospital", value="Go to the nearest emergency room — do not wait for a callback."),
    EmergencyResource(label="Note", value="This is a prototype and cannot verify real-time hospital or helpline availability."),
]


@dataclass
class PipelineResult:
    """Everything both the API response and the audit log need — the cacheable unit.
    Deliberately independent of the AskResponse Pydantic model so the cache isn't
    coupled to the API schema.
    """

    status: str  # "answered" | "refused"
    refusal_reason: RefusalReason | None = None
    answer: str | None = None
    citations: list[Citation] = field(default_factory=list)
    confidence_score: float | None = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    ungrounded_count: int = 0
    total_sentences: int = 0


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    user: User = Depends(require_agent),
    db: Session = Depends(get_db),
) -> AskResponse:
    settings = get_settings()

    # The emergency check always runs fresh, on every request, before the cache OR the
    # rate limiter is ever consulted — a cache hit must never be how a query skips this
    # gate, and a rate-limited user must still get an emergency card, not a 429, for a
    # chest-pain query. This is also why emergency results are never written to the
    # cache below: the check is a near-free regex, not what caching is for here, and
    # this keeps the safety-critical path entirely outside both mechanisms' reach.
    emergency = check_emergency(request.query)
    if emergency.flagged:
        _log(db, user, request.query, [], None, refused=True, reason=RefusalReason.emergency_flagged)
        return AskResponse(status="emergency", refusal_reason="emergency_flagged", emergency_resources=EMERGENCY_RESOURCES)

    enforce_ask_rate_limit(user.id)

    result = query_cache.get(request.query)
    if result is None:
        result = await _run_pipeline(db, settings, request.query)
        query_cache.set(request.query, result)

    _log(db, user, request.query, result.retrieved_chunk_ids, result.confidence_score, refused=(result.status != "answered"), reason=result.refusal_reason, ungrounded=result.ungrounded_count)

    return AskResponse(
        status=result.status,
        answer=result.answer,
        citations=result.citations,
        confidence_score=result.confidence_score,
        refusal_reason=result.refusal_reason.value if result.refusal_reason else None,
    )


async def _run_pipeline(db: Session, settings: Settings, query: str) -> PipelineResult:
    # Expansion bridges brand names and colloquialisms ("Advil", "blood thinner", "Dolo")
    # to the generic-name vocabulary the corpus is written in. It only ever widens what
    # retrieval and reranking search for — the LLM prompt below still uses the original
    # `query`, so the answer is phrased against what was actually asked.
    retrieval_query = expand_query(query)

    retriever = registry.get_retriever()
    fused = retriever.retrieve(retrieval_query, settings.top_k_candidates, settings.top_k_candidates)

    if not fused:
        return PipelineResult(status="refused", refusal_reason=RefusalReason.out_of_scope)

    chunk_rows = (
        db.query(Chunk, Document.title)
        .join(Document, Chunk.doc_id == Document.id)
        .filter(Chunk.id.in_([c.chunk_id for c in fused]))
        .all()
    )
    chunk_lookup = {
        chunk.id: {"chunk_id": chunk.id, "content": chunk.content, "page_number": chunk.page_number, "document_title": title}
        for chunk, title in chunk_rows
    }
    candidates = [chunk_lookup[c.chunk_id] for c in fused if c.chunk_id in chunk_lookup]

    reranked = rerank(retrieval_query, candidates, settings.top_k_reranked)
    confidence = compute_confidence(reranked, fused)
    retrieved_chunk_ids = [r.chunk_id for r in reranked]

    if confidence.score < settings.confidence_threshold:
        return PipelineResult(
            status="refused",
            refusal_reason=RefusalReason.low_confidence,
            confidence_score=confidence.score,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

    prompt = build_prompt(query, reranked)
    answer_text = await generate(prompt)

    groundedness = check_groundedness(answer_text, [r.content for r in reranked])
    final_answer = " ".join(s.sentence for s in groundedness.grounded_sentences)

    if not final_answer.strip():
        return PipelineResult(
            status="refused",
            refusal_reason=RefusalReason.low_confidence,
            confidence_score=confidence.score,
            retrieved_chunk_ids=retrieved_chunk_ids,
            ungrounded_count=groundedness.ungrounded_count,
            total_sentences=groundedness.total_sentences,
        )

    # Citations reflect what the surviving answer is actually grounded in — the
    # chunk each grounded sentence was closest to — not what the model happened to
    # write in a citation marker (see generation/prompt.py).
    cited_chunk_indices = dict.fromkeys(s.best_chunk_index for s in groundedness.grounded_sentences)
    citations = [
        Citation(document_title=reranked[i].document_title, page_number=reranked[i].page_number, chunk_id=reranked[i].chunk_id)
        for i in cited_chunk_indices
        if i is not None
    ]

    return PipelineResult(
        status="answered",
        answer=final_answer,
        citations=citations,
        confidence_score=confidence.score,
        retrieved_chunk_ids=retrieved_chunk_ids,
        ungrounded_count=groundedness.ungrounded_count,
        total_sentences=groundedness.total_sentences,
    )


def _log(
    db: Session,
    user: User,
    query_text: str,
    retrieved_chunk_ids: list[str],
    confidence_score: float | None,
    refused: bool,
    reason: RefusalReason | None = None,
    ungrounded: int = 0,
) -> None:
    db.add(
        QueryLog(
            user_id=user.id,
            query_text=query_text,
            retrieved_chunk_ids=retrieved_chunk_ids,
            confidence_score=confidence_score,
            was_refused=refused,
            refusal_reason=reason,
            ungrounded_claim_count=ungrounded,
        )
    )
    db.commit()
