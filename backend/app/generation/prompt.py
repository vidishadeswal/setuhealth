"""Context-restricted prompt construction (design doc Section 11): the model only ever
sees the top reranked chunks plus the question — never open-domain latitude — and is
required to cite (document, page) for every claim.

The citation instruction below is for the *prose* — it makes the answer readable and
self-checkable against the passages. The citations actually returned to the caller are
derived separately, from embedding-similarity evidence (safety/groundedness.py), not by
parsing these markers back out: a small local model won't always format them exactly as
asked, and trusting free-text formatting for the one thing this product is supposed to
guarantee (which sources back this answer) is exactly the fragility SetuHealth exists to
avoid.
"""

from backend.app.retrieval.reranker import RerankedChunk

SYSTEM_INSTRUCTION = """You are answering a drug-interaction question using ONLY the passages below. \
Do not use any knowledge outside them, even if you believe it to be true.

For every claim, cite the source in this exact form: (Source: {document title}, p.{page number})

If the passages do not fully answer the question, say exactly what is missing instead of guessing. \
Do not hedge with disclaimers about consulting a doctor — that is handled separately by the system, \
not part of your answer."""


def build_prompt(query: str, chunks: list[RerankedChunk]) -> str:
    passages = "\n\n".join(
        f"[Passage {i + 1}] (Source: {c.document_title}, p.{c.page_number})\n{c.content}"
        for i, c in enumerate(chunks)
    )
    return f"{SYSTEM_INSTRUCTION}\n\nPASSAGES:\n{passages}\n\nQUESTION: {query}\n\nANSWER:"
