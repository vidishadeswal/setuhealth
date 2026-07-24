"""LLM-assisted retrieval fallback: multi-query rewriting + HyDE (Hypothetical Document
Embeddings). Used only when the cheap path (query_expansion's alias table) isn't enough —
see api/routes_ask.py, which calls this only when first-pass confidence is already below
threshold. Two established, training-free techniques, combined in one call:

1. Multi-query: paraphrase the question in different clinical/lay vocabulary, so BM25 and
   the embedding search get more surface area to match against than one fixed phrasing.
2. HyDE: ask the model for a short hypothetical *statement* (not a question) in the
   register an FDA label actually uses. Motivation is empirical, not theoretical — reranker
   debugging during development (see safety/confidence.py's calibration notes) showed the
   cross-encoder consistently scores label-style declarative text far higher against real
   label passages than a natural question does, even for the identical passage.

The hypothetical statement is invented by the model and may be factually wrong — expected,
and harmless on its own, since it's used only to steer retrieval and is never shown to the
user. What is NOT harmless: a 3B local model asked to always produce a HyDE sentence will
also fabricate one for a question that has nothing to do with drug interactions at all,
and that fabrication can accidentally retrieve real corpus text with deceptively high
confidence. An explicit "is this even in scope" instruction in the prompt did not fix this
reliably in testing — the model doesn't self-police well. The actual safeguard is
structural, not prompt-level: see has_lexical_overlap() below and its caller in
api/routes_ask.py, which refuses to accept a rewrite-assisted result unless the *real*
query (not a rewrite) shares real vocabulary with the chunk it supposedly justified.
"""

import re

from backend.app.generation.llm_client import generate

_LINE_RE = re.compile(r"^(REWRITE|HYDE):\s*(.+)$", re.IGNORECASE)

PROMPT_TEMPLATE = """You are helping a search system find the right passage for a drug-interaction \
question. You are NOT answering the question.

Given the question below, produce exactly these three lines and nothing else:
REWRITE: the same question, with every brand name or colloquial drug term replaced by its \
generic pharmaceutical name (example: "Advil" becomes "ibuprofen", "blood thinner" becomes \
"warfarin" or "anticoagulant", "Tylenol" or "Dolo" becomes "acetaminophen")
REWRITE: the same question rewritten as a plain clinical statement instead of a question \
(example: "Can I take X with Y?" becomes "Interaction between X and Y")
HYDE: one short hypothetical sentence, in the plain declarative style of an FDA drug \
label's drug-interactions section, that might plausibly appear in an answer to this \
question — it does not need to be verified or even true, it is only used to help a search \
index find similar real text

Question: {query}"""

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "i", "you", "he", "she", "it", "we", "they", "of", "in", "on", "with", "for", "to",
    "and", "or", "not", "be", "if", "at", "by", "as", "my", "me", "this", "that",
    "what", "how", "when", "why", "who", "which", "take", "taking", "while",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def has_lexical_overlap(query: str, text: str) -> bool:
    """True if the query shares at least one meaningful (non-stopword) token with text.
    The gate that makes the rewrite fallback safe: a fabricated HyDE sentence or a
    garbled rewrite can score a completely unrelated chunk highly, but it can't make
    the *real* query's own words appear in that chunk's actual content. Deliberately as
    dumb as the emergency heuristic — auditable beats clever for a safety-relevant gate.
    """
    query_tokens = {w for w in _WORD_RE.findall(query.lower()) if w not in _STOPWORDS and len(w) > 2}
    text_lower = text.lower()
    return any(token in text_lower for token in query_tokens)


def _parse_variants(raw: str) -> list[str]:
    variants = []
    for line in raw.splitlines():
        match = _LINE_RE.match(line.strip())
        if match:
            text = match.group(2).strip()
            if text:
                variants.append(text)
    return variants


async def generate_retrieval_variants(query: str) -> list[str]:
    """Returns 0-3 additional query strings to retrieve with, alongside the original.
    Never raises on malformed model output or a failed LLM call — worst case, returns an
    empty list and the caller just retries with what it already had.
    """
    try:
        raw = await generate(PROMPT_TEMPLATE.format(query=query), temperature=0.15)
    except Exception:
        return []

    return _parse_variants(raw)
