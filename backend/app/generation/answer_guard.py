"""Deterministic post-generation checks on wording the prompt can only request, not
guarantee. A 3B local model still says "contraindicated" for labels that only say
"generally should not be given" (seen live: furosemide + lithium) — a stronger claim
than the source makes. Prompt rules reduced this but can't eliminate it, so the one
severity word that matters most is checked in code against the actual passages.
"""

import re

_CONTRA_RE = re.compile(r"\bcontraindicated\b", re.IGNORECASE)


def enforce_severity_wording(answer: str, passages: list[str]) -> str:
    """Replace "contraindicated" with "not recommended" unless at least one retrieved
    passage itself uses the word (contraindicated / contraindication)."""
    if not _CONTRA_RE.search(answer):
        return answer
    if any("contraindicat" in p.lower() for p in passages):
        return answer
    return _CONTRA_RE.sub("not recommended", answer)


_NO_INTERACTION_RE = re.compile(r"no interaction between these two is noted in the passages", re.IGNORECASE)


def drop_self_contradiction(answer: str) -> str:
    """The prompt's fallback sentence "No interaction between these two is noted in the
    passages." sometimes gets appended by the small model AFTER it has already stated an
    interaction (seen live: sertraline + tramadol). Both can't be true of the same pair,
    so when other content exists, the blanket denial is dropped; a lone denial is kept.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", answer.strip()) if p.strip()]
    kept = [p for p in parts if not _NO_INTERACTION_RE.search(p)]
    if not kept or len(kept) == len(parts):
        return answer
    return " ".join(kept)
