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
from backend.app.generation.query_rewrite import generate_retrieval_variants, has_lexical_overlap
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.models.query_log import QueryLog, RefusalReason
from backend.app.models.user import User
from backend.app.retrieval.hybrid import FusedCandidate, HybridRetriever
from backend.app.retrieval.query_expansion import expand_query
from backend.app.retrieval.registry import registry
from backend.app.retrieval.reranker import RerankedChunk, rerank_multi
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
    used_query_rewrite: bool = False


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


def _retrieve_and_rerank(
    db: Session, settings: Settings, retriever: HybridRetriever, queries: list[str]
) -> tuple[list[RerankedChunk], list[FusedCandidate]]:
    """Retrieval + reranking for one or more query variants. Both stages see the same
    variant list: retrieval casts as wide a net as the variants allow, and reranking
    scores each candidate against every variant and keeps its best score — a paraphrase
    that widens retrieval but never reaches reranking wouldn't help, since a passage
    that's genuinely relevant only needs to score well under ONE phrasing to prove it.
    """
    fused = retriever.retrieve_multi(queries, settings.top_k_candidates, settings.top_k_candidates)
    if not fused:
        return [], fused

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
    reranked = rerank_multi(queries, candidates, settings.top_k_reranked)
    return reranked, fused


async def _run_pipeline(db: Session, settings: Settings, query: str) -> PipelineResult:
    # Expansion bridges brand names and colloquialisms ("Advil", "blood thinner", "Dolo")
    # to the generic-name vocabulary the corpus is written in. It only ever widens what
    # retrieval and reranking search for — the LLM prompt below still uses the original
    # `query`, so the answer is phrased against what was actually asked.
    retrieval_query = expand_query(query)
    retriever = registry.get_retriever()

    reranked, fused = _retrieve_and_rerank(db, settings, retriever, [retrieval_query])
    if not fused:
        return PipelineResult(status="refused", refusal_reason=RefusalReason.out_of_scope)
    confidence = compute_confidence(reranked, fused)

    # The cheap path (BM25 + vector + alias expansion) already answers most questions.
    # Only when it isn't enough do we pay for an extra LLM call: ask the model itself for
    # paraphrases and a HyDE-style hypothetical passage, retrieve again with that wider
    # net, and keep whichever attempt scored higher. This targets the fallback's latency
    # cost at exactly the hard cases, instead of taxing every request for it.
    used_query_rewrite = False
    if confidence.score < settings.confidence_threshold:
        variants = await generate_retrieval_variants(query)
        if variants:
            retry_reranked, retry_fused = _retrieve_and_rerank(
                db, settings, retriever, [retrieval_query, *variants]
            )
            if retry_fused and retry_reranked:
                retry_confidence = compute_confidence(retry_reranked, retry_fused)
                # The overlap check is the actual safety gate here, not the confidence
                # comparison alone: a fabricated HyDE sentence for a genuinely unrelated
                # question (e.g. "What is the capital of France?") can score some real
                # chunk highly on its own — see generation/query_rewrite.py's docstring.
                # Requiring the REAL query to share real vocabulary with the winning
                # chunk is what stops that from ever being accepted as an answer.
                improved = retry_confidence.score > confidence.score
                grounded_in_real_query = has_lexical_overlap(retrieval_query, retry_reranked[0].content)
                if improved and grounded_in_real_query:
                    reranked, fused, confidence = retry_reranked, retry_fused, retry_confidence
                    used_query_rewrite = True

    retrieved_chunk_ids = [r.chunk_id for r in reranked]

    if confidence.score < settings.confidence_threshold:
        return PipelineResult(
            status="refused",
            refusal_reason=RefusalReason.low_confidence,
            confidence_score=confidence.score,
            retrieved_chunk_ids=retrieved_chunk_ids,
            used_query_rewrite=used_query_rewrite,
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
            used_query_rewrite=used_query_rewrite,
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
        used_query_rewrite=used_query_rewrite,
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
