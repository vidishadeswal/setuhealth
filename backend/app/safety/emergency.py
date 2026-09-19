"""Deterministic emergency heuristic (design doc Section 12) — runs before retrieval and
before any LLM call. Hand-tuned keyword/pattern matching, not a model call: brittle but
auditable and directly red-teamable (see eval/redteam_eval.jsonl), which matters more
than sophistication on the one gate where a false negative is a real-harm event.

Hypothetical/indirect framing ("hypothetically, if someone had chest pain...") is not
specially parsed — the symptom keywords are still literally present in the string and
still match. The real evasion risk is a symptom described without any of these words at
all, which is exactly what the red-team eval set is designed to surface.
"""

import re
from dataclasses import dataclass

# Each category is a list of regex patterns, not single keywords, to catch common
# phrasings without requiring an exact match.
EMERGENCY_PATTERNS: dict[str, list[str]] = {
    "chest_pain": [
        r"\bchest pain\b", r"\bchest (is |feels? )?(tight|tightness|crushing|heavy)\b",
        r"\bpain (in|across) (my |the |his |her )?chest\b",
        # Symptom described without the word "pain": "chest just feels weird and tight",
        # "pressure on my chest", "squeezing in my chest".
        r"\bchest\b.{0,30}\b(tight|tightness|pressure|crushing|heavy|squeez\w*|weird|funny)\b",
        r"\b(pressure|squeezing|tightness|weight) (on|in|across) (my |the |his |her )?chest\b",
        r"\b(sitting|standing|pressing|weighing|resting) (down )?on (my |the |his |her )?chest\b",
    ],
    "breathing_difficulty": [
        r"\b(can'?t|cannot|difficulty|struggling to|trouble) breath",
        r"\bshort(ness)? of breath\b", r"\bgasping for air\b", r"\bthroat (is )?closing\b",
        # Indirect phrasings: "hard to get air in", "can't get enough air", "breathing is hard".
        r"\b(hard|difficult|impossible) to (get|catch|take|draw) (any |enough )?(air|a breath|my breath|breath)",
        r"\b(can'?t|cannot|struggling to|unable to|trouble trying to) (get|catch) (any |enough )?(air|my breath|a breath)\b",
        r"\bbreathing (is |has (gotten|become) )?(really |very )?(hard|difficult|labou?red)\b",
    ],
    "severe_bleeding": [
        r"\b(severe|heavy|won'?t stop|uncontrolled) bleeding\b", r"\bbleeding (won'?t|does not|doesn'?t) stop\b",
        r"\bhemorrhag(e|ing)\b",
    ],
    "anaphylaxis": [
        r"\banaphylaxis\b", r"\ballergic reaction.*(swelling|throat|breath)",
        r"\b(face|lips|tongue|throat) (is |are )?swelling\b",
    ],
    "suicidal_ideation": [
        r"\bsuicid", r"\b(want|going|plan) to (kill myself|end (my|it) (life|all))\b",
        r"\bself[- ]harm\b", r"\bharm(ing)? myself\b",
    ],
    "overdose": [
        r"\bover ?dose", r"\btook too many (pills|tablets|tabs)\b", r"\bswallowed (a lot|too many)\b",
    ],
    "stroke_signs": [
        r"\bface (is )?droop", r"\bslurred speech\b", r"\bsudden (weakness|numbness)\b.*\b(arm|face|leg|side)\b",
        r"\bcan'?t (move|feel) (my|one) (arm|leg|side)\b",
    ],
    "unresponsive": [
        r"\bunresponsive\b", r"\bunconscious\b", r"\bnot waking up\b", r"\bpassed out\b.*\b(not|won'?t) (waking|responding)\b",
    ],
    "pregnancy_complication": [
        r"\b(pregnant|pregnancy).*(severe|heavy) (pain|bleeding)\b",
        r"\b(severe|heavy) bleeding.*\bpregnan",
    ],
}

_COMPILED = {cat: [re.compile(p, re.IGNORECASE) for p in patterns] for cat, patterns in EMERGENCY_PATTERNS.items()}


@dataclass
class EmergencyCheckResult:
    flagged: bool
    matched_category: str | None = None
    matched_text: str | None = None


def check_emergency(query_text: str) -> EmergencyCheckResult:
    for category, patterns in _COMPILED.items():
        for pattern in patterns:
            match = pattern.search(query_text)
            if match:
                return EmergencyCheckResult(flagged=True, matched_category=category, matched_text=match.group(0))
    return EmergencyCheckResult(flagged=False)
