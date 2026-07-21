"""Query expansion for retrieval: bridges the vocabulary gap between how people actually
ask ("Advil", "blood thinner", "Dolo") and the generic-name vocabulary FDA labels use
("ibuprofen", "warfarin", "acetaminophen"). Chosen over fine-tuning an embedding model —
that would need labeled data, GPU compute, and retraining for every corpus change; this
is a flat table anyone can extend in one line, and its effect on any given query is fully
inspectable. Same reasoning as the emergency heuristic in safety/emergency.py: boring and
auditable beats sophisticated and opaque for a component this load-bearing.

This expands the text used for retrieval and reranking only. The LLM prompt and anything
shown back to the user always use the caller's original phrasing — expansion is a
retrieval aid, not a rewrite of what was asked.
"""

import re

# colloquial term or brand name -> canonical generic name(s) as they appear in the corpus.
# Multi-word keys are matched as phrases. Extend this table as the corpus grows.
DRUG_ALIASES: dict[str, list[str]] = {
    # Warfarin
    "coumadin": ["warfarin"],
    "jantoven": ["warfarin"],
    "blood thinner": ["warfarin", "anticoagulant"],
    "blood thinners": ["warfarin", "anticoagulant"],
    # Ibuprofen
    "advil": ["ibuprofen"],
    "motrin": ["ibuprofen"],
    "nurofen": ["ibuprofen"],
    "brufen": ["ibuprofen"],
    # Acetaminophen / paracetamol — both terms are cross-referenced since either may be
    # the query term or the corpus term depending on the user's region.
    "paracetamol": ["acetaminophen"],
    "acetaminophen": ["paracetamol"],
    "tylenol": ["acetaminophen", "paracetamol"],
    "dolo": ["acetaminophen", "paracetamol"],
    "dolo 650": ["acetaminophen", "paracetamol"],
    "crocin": ["acetaminophen", "paracetamol"],
    "calpol": ["acetaminophen", "paracetamol"],
    "panadol": ["acetaminophen", "paracetamol"],
    # Doxycycline
    "vibramycin": ["doxycycline"],
    "doryx": ["doxycycline"],
    "monodox": ["doxycycline"],
    # Ciprofloxacin
    "cipro": ["ciprofloxacin"],
    # Sertraline
    "zoloft": ["sertraline"],
    # Simvastatin
    "zocor": ["simvastatin"],
}

# Longest terms first so "dolo 650" is matched as a phrase before the bare "dolo" inside
# it competes for the same starting position.
_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(DRUG_ALIASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def expand_query(query: str) -> str:
    """Appends canonical generic-name terms for any recognized alias found in the query.
    Retrieval-only — never shown to the user or passed to the LLM as the question asked.
    """
    matched_terms: list[str] = []
    for match in _PATTERN.finditer(query):
        matched_terms.extend(DRUG_ALIASES[match.group(0).lower()])

    if not matched_terms:
        return query

    unique_terms = dict.fromkeys(matched_terms)  # de-dup, preserve first-seen order
    return f"{query} {' '.join(unique_terms)}"
