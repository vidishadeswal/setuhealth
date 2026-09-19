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

from backend.app.generation.table_reconstruct import reconstruct_tables
from backend.app.retrieval.reranker import RerankedChunk

SYSTEM_INSTRUCTION = """You are answering a drug-interaction question using ONLY the passages below. \
Do not use any knowledge outside them, even if you believe it to be true.

Start your answer with a single-sentence bottom line stating what the passages say about this \
specific combination. Match the severity word to what the passages actually say — do not default \
to a stronger or more alarming word than the passages use:
- Say "contraindicated" ONLY if a passage explicitly states the combination should not be used, \
is contraindicated, or must be avoided entirely.
- If a passage says "should not be given", "avoid", or "not recommended" without also saying \
contraindicated, say the combination "should generally be avoided" — that is NOT the same as \
contraindicated either.
- If a passage says to monitor, use caution, or that risk is increased, say exactly that: state \
that the risk is increased and name the specific risk the passage itself names, or say it is \
generally safe but should be monitored and name what to monitor. This is NOT the same as \
contraindicated, and is the most common case in this corpus. Never write a placeholder letter \
such as "X" — always name the real thing from the passage.
- If nothing in the passages addresses this specific combination, say so plainly: "No interaction \
between these two is noted in the passages."
Do not open with a hedge or disclaimer before this sentence — if the passages contain information \
relevant to the question, lead with it, then explain. Only say information is missing after first \
stating whatever the passages do establish, and only for the specific part that is genuinely absent \
— not as a substitute for a bottom line the passages actually support.

For every claim, cite the source in this exact form: (Source: {document title}, p.{page number})

Do not hedge with disclaimers about consulting a doctor — that is handled separately by the system, \
not part of your answer."""


def build_prompt(query: str, chunks: list[RerankedChunk], alias_hints: dict[str, list[str]] | None = None) -> str:
    # Chunk content has its original line breaks collapsed to single spaces (see
    # ingestion/chunker.py). For a passage that's actually a table, that flattening
    # measurably causes value/row misattribution in generation — reconstruct real
    # newlines here so the model reads one row per line.
    passages = "\n\n".join(
        f"[Passage {i + 1}] (Source: {c.document_title}, p.{c.page_number})\n{reconstruct_tables(c.content)}"
        for i, c in enumerate(chunks)
    )

    # alias_hints comes from query_expansion.find_aliases() — the same lookup table
    # already trusted to find these passages in the first place, not an LLM guess. Retrieval
    # can find a passage about "acetaminophen" for a question about "Dolo", but the model
    # has no way to connect the two unless told: it only ever sees the user's original
    # phrasing, and the passages use whatever vocabulary the source document does.
    hint_block = ""
    if alias_hints:
        lines = "\n".join(f"- \"{term}\" refers to: {', '.join(canonicals)}" for term, canonicals in alias_hints.items())
        hint_block = f"\n\nThe question may use brand names or informal terms. Known mappings:\n{lines}"

    return f"{SYSTEM_INSTRUCTION}{hint_block}\n\nPASSAGES:\n{passages}\n\nQUESTION: {query}\n\nANSWER:"
