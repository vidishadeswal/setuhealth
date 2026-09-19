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

Two tiers, kept deliberately separate:
  - DRUG_ALIASES below: hand-curated, individually reviewed, small.
  - data/brand_aliases.json: ~80 real Indian brand names, bulk-extracted from a public
    Kaggle dataset (scripts/extract_brand_aliases.py) for the corpus's covered generics
    only. Unverified provenance — that's fine here specifically because this data never
    becomes cited content or grounds an answer; it only ever widens what a query is
    allowed to match against, same as the hand-curated table. If provenance mattered for
    what this data is used for, it would not be here — see corpus/sample/ for the bar
    that applies to anything actually cited to a user.
"""

import difflib
import json
import re
from pathlib import Path

# The corpus's actual generic drug names — used both as fuzzy-match targets below (a
# misspelled generic name needs correcting too, not just misspelled brand names) and as
# documentation of what this whole alias system is ultimately resolving toward.
CORPUS_GENERICS = [
    "warfarin", "doxycycline", "ciprofloxacin", "sertraline", "simvastatin", "acetaminophen",
    "ibuprofen", "tramadol", "clopidogrel", "digoxin", "methotrexate", "phenytoin", "cyclosporine",
    "metformin", "atorvastatin", "lisinopril", "omeprazole", "levothyroxine", "amoxicillin",
    "azithromycin", "fluoxetine", "amlodipine",
    "losartan", "hydrochlorothiazide", "prednisone", "gabapentin", "citalopram", "furosemide",
    "montelukast", "pantoprazole", "alprazolam", "metoprolol", "spironolactone", "allopurinol",
    "duloxetine", "escitalopram", "tamsulosin",
]

# colloquial term or brand name -> canonical generic name(s) as they appear in the corpus.
# Multi-word keys are matched as phrases. Extend this table as the corpus grows.
DRUG_ALIASES: dict[str, list[str]] = {
    # Warfarin and clopidogrel — both are colloquially called "blood thinners" by
    # patients even though pharmacologically warfarin is an anticoagulant and
    # clopidogrel is an antiplatelet. Expanding to both rather than picking one avoids
    # silently missing the drug the patient actually means.
    "coumadin": ["warfarin"],
    "jantoven": ["warfarin"],
    "blood thinner": ["warfarin", "clopidogrel", "anticoagulant"],
    "blood thinners": ["warfarin", "clopidogrel", "anticoagulant"],
    # Ibuprofen
    "advil": ["ibuprofen"],
    "motrin": ["ibuprofen"],
    "nurofen": ["ibuprofen"],
    "brufen": ["ibuprofen"],
    # Tramadol
    "ultram": ["tramadol"],
    "conzip": ["tramadol"],
    # Clopidogrel
    "plavix": ["clopidogrel"],
    # Digoxin
    "lanoxin": ["digoxin"],
    # Methotrexate
    "trexall": ["methotrexate"],
    "otrexup": ["methotrexate"],
    # Phenytoin and gabapentin are both anticonvulsants in this corpus now.
    "dilantin": ["phenytoin"],
    "anticonvulsant": ["phenytoin", "gabapentin"],
    "seizure medicine": ["phenytoin", "gabapentin"],
    # Cyclosporine
    "neoral": ["cyclosporine"],
    "sandimmune": ["cyclosporine"],
    "gengraf": ["cyclosporine"],
    "immunosuppressant": ["cyclosporine"],
    # Metformin
    "glucophage": ["metformin"],
    "fortamet": ["metformin"],
    "glumetza": ["metformin"],
    "diabetes medicine": ["metformin"],
    "diabetes pill": ["metformin"],
    "sugar medicine": ["metformin"],
    # Atorvastatin
    "lipitor": ["atorvastatin"],
    # Lisinopril, amlodipine, losartan, hydrochlorothiazide, and metoprolol all treat
    # blood pressure, so a colloquial "bp medicine" reference expands to all five
    # rather than guessing one.
    "prinivil": ["lisinopril"],
    "zestril": ["lisinopril"],
    "ace inhibitor": ["lisinopril"],
    "blood pressure medicine": ["lisinopril", "amlodipine", "losartan", "hydrochlorothiazide", "metoprolol"],
    "bp medicine": ["lisinopril", "amlodipine", "losartan", "hydrochlorothiazide", "metoprolol"],
    "bp tablet": ["lisinopril", "amlodipine", "losartan", "hydrochlorothiazide", "metoprolol"],
    # Omeprazole and pantoprazole are both PPIs in this corpus now.
    "prilosec": ["omeprazole"],
    "losec": ["omeprazole"],
    "acid reducer": ["omeprazole", "pantoprazole"],
    "heartburn medicine": ["omeprazole", "pantoprazole"],
    "ppi": ["omeprazole", "pantoprazole"],
    # Losartan
    "cozaar": ["losartan"],
    "arb": ["losartan"],
    "angiotensin receptor blocker": ["losartan"],
    # Hydrochlorothiazide, furosemide, and spironolactone are all diuretics — expand
    # "water pill" to all three rather than guessing which one.
    "microzide": ["hydrochlorothiazide"],
    "hctz": ["hydrochlorothiazide"],
    "water pill": ["hydrochlorothiazide", "furosemide", "spironolactone"],
    "diuretic": ["hydrochlorothiazide", "furosemide", "spironolactone"],
    # Prednisone — only steroid in this corpus, safe to alias unambiguously.
    "deltasone": ["prednisone"],
    "rayos": ["prednisone"],
    "steroid": ["prednisone"],
    "steroids": ["prednisone"],
    # Gabapentin
    "neurontin": ["gabapentin"],
    "nerve pain medicine": ["gabapentin"],
    # Citalopram
    "celexa": ["citalopram"],
    # Furosemide
    "lasix": ["furosemide"],
    # Montelukast
    "singulair": ["montelukast"],
    # Pantoprazole
    "protonix": ["pantoprazole"],
    # Alprazolam
    "xanax": ["alprazolam"],
    "benzodiazepine": ["alprazolam"],
    "benzo": ["alprazolam"],
    # Metoprolol
    "lopressor": ["metoprolol"],
    "toprol": ["metoprolol"],
    "beta blocker": ["metoprolol"],
    # Spironolactone
    "aldactone": ["spironolactone"],
    # Allopurinol — only gout drug in this corpus, safe to alias unambiguously.
    "zyloprim": ["allopurinol"],
    "gout medicine": ["allopurinol"],
    # Duloxetine
    "cymbalta": ["duloxetine"],
    "snri": ["duloxetine"],
    # Escitalopram
    "lexapro": ["escitalopram"],
    # Tamsulosin
    "flomax": ["tamsulosin"],
    # Levothyroxine
    "synthroid": ["levothyroxine"],
    "levoxyl": ["levothyroxine"],
    "euthyrox": ["levothyroxine"],
    "thyroid medicine": ["levothyroxine"],
    "thyroid pill": ["levothyroxine"],
    # Amoxicillin
    "amoxil": ["amoxicillin"],
    "trimox": ["amoxicillin"],
    # Azithromycin
    "zithromax": ["azithromycin"],
    "z-pack": ["azithromycin"],
    "zpack": ["azithromycin"],
    "zmax": ["azithromycin"],
    # Fluoxetine
    "prozac": ["fluoxetine"],
    "sarafem": ["fluoxetine"],
    # Amlodipine
    "norvasc": ["amlodipine"],
    "calcium channel blocker": ["amlodipine"],
    # Class-vocabulary bridges: labels often name the CLASS, not the specific drug a user
    # types ("oral anticoagulants", "PDE5 inhibitors", "digitalis glycosides"), so a query
    # naming the drug never lexically meets the passage that answers it. Found by eval:
    # "levothyroxine + warfarin" missed levothyroxine's "Oral Anticoagulants" section, and
    # "tamsulosin + sildenafil" missed its "PDE5 Inhibitors" section.
    "warfarin": ["warfarin", "oral anticoagulants", "coumarin"],
    "digoxin": ["digoxin", "digitalis glycosides"],
    "sildenafil": ["PDE5 inhibitors"],
    "tadalafil": ["PDE5 inhibitors"],
    "vardenafil": ["PDE5 inhibitors"],
    "viagra": ["sildenafil", "PDE5 inhibitors"],
    "cialis": ["tadalafil", "PDE5 inhibitors"],
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
    # Sertraline, fluoxetine, citalopram, and escitalopram are all SSRIs in this corpus,
    # so an indication-level reference ("my antidepressant") can't be pinned to one —
    # expand to all of them (plus duloxetine, an SNRI, for the broader "antidepressant"
    # term specifically) and let retrieval/reranking sort out which passage matches.
    # Brand names stay unambiguous since they each name one specific drug.
    "zoloft": ["sertraline"],
    "antidepressant": ["sertraline", "fluoxetine", "citalopram", "escitalopram", "duloxetine"],
    "my antidepressant": ["sertraline", "fluoxetine", "citalopram", "escitalopram", "duloxetine"],
    "ssri": ["sertraline", "fluoxetine", "citalopram", "escitalopram"],
    # Simvastatin and atorvastatin are both statins in this corpus now — same reasoning.
    "zocor": ["simvastatin"],
    "statin": ["simvastatin", "atorvastatin"],
    "cholesterol medicine": ["simvastatin", "atorvastatin"],
    "cholesterol medication": ["simvastatin", "atorvastatin"],
    "cholesterol pill": ["simvastatin", "atorvastatin"],
}

_GENERATED_ALIASES_PATH = Path(__file__).parent / "data" / "brand_aliases.json"


def _load_generated_aliases() -> dict[str, list[str]]:
    if not _GENERATED_ALIASES_PATH.exists():
        return {}
    with open(_GENERATED_ALIASES_PATH, encoding="utf-8") as f:
        return json.load(f)["aliases"]


def _merge_aliases(*tables: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for table in tables:
        for key, canonicals in table.items():
            existing = merged.setdefault(key, [])
            for c in canonicals:
                if c not in existing:
                    existing.append(c)
    return merged


ALL_ALIASES: dict[str, list[str]] = _merge_aliases(DRUG_ALIASES, _load_generated_aliases())

# Longest terms first so "dolo 650" is matched as a phrase before the bare "dolo" inside
# it competes for the same starting position.
_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in sorted(ALL_ALIASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# Fuzzy-match targets: single-word alias keys plus the corpus's own generic names — a
# typo'd generic name ("warfrin") needs correcting just as much as a typo'd brand name.
# Multi-word keys (e.g. "blood thinner") are excluded: fuzzy-matching whole phrases
# against single mistyped words is a different, noisier problem than this is meant to
# solve.
_FUZZY_TARGETS: dict[str, list[str]] = {
    **{k: v for k, v in ALL_ALIASES.items() if " " not in k},
    **{g: [g] for g in CORPUS_GENERICS},
}
_WORD_RE = re.compile(r"[a-zA-Z]+")
_FUZZY_MIN_LENGTH = 5  # shorter words risk too many false-positive "close" matches
_FUZZY_CUTOFF = 0.82  # difflib similarity ratio; ~1 edit on a 6-8 letter drug name

# Real medical words that happen to sit within one edit of an unrelated bulk-extracted
# brand name and would otherwise get silently "corrected" to the wrong drug. Found
# live: "lithium" (a real drug named throughout many interaction passages, e.g.
# furosemide's and digoxin's own labels) is one edit from "zithium", a generated brand
# alias for azithromycin — a query naming lithium was silently expanded toward an
# unrelated antibiotic. Add here, don't just raise the cutoff globally: a stricter
# cutoff would also block legitimate catches like "warfrin" -> warfarin.
_FUZZY_PROTECTED_WORDS = {"lithium", "thyroid"}


def _fuzzy_matches(query: str, already_matched: set[str]) -> dict[str, list[str]]:
    """Catches simple misspellings of drug names ("warfrin" -> warfarin) that exact
    alias matching can't. Real users mistype drug names constantly, especially on
    mobile. Deliberately conservative — long words only, high similarity cutoff — same
    "boring beats clever" bar as the rest of this module: a wrong fuzzy correction would
    silently misdirect retrieval, so it's better to miss a typo than guess one wrong.
    """
    corrections: dict[str, list[str]] = {}
    for word in _WORD_RE.findall(query):
        lower = word.lower()
        if (
            len(lower) < _FUZZY_MIN_LENGTH
            or lower in already_matched
            or lower in _FUZZY_TARGETS
            or lower in _FUZZY_PROTECTED_WORDS
        ):
            continue
        close = difflib.get_close_matches(lower, _FUZZY_TARGETS, n=1, cutoff=_FUZZY_CUTOFF)
        if close:
            corrections[word] = _FUZZY_TARGETS[close[0]]
    return corrections


def find_aliases(query: str) -> dict[str, list[str]]:
    """Returns {matched term (as written in the query): [canonical generic name(s)]}
    for every alias recognized in the query, preserving first-seen order — exact alias
    matches first, then fuzzy-corrected typos of drug names not otherwise matched.
    Exposed separately from expand_query() so callers that need to know *which* terms
    resolved to *what* — not just the retrieval-query string — can use it too. See
    generation/prompt.py: the LLM only ever sees the user's original phrasing, so if a
    query says "Dolo" and the only relevant passage is about "acetaminophen", the model
    has no way to make that connection unless it's told explicitly.
    """
    matches: dict[str, list[str]] = {}
    for match in _PATTERN.finditer(query):
        matched_text = match.group(0)
        matches[matched_text] = ALL_ALIASES[matched_text.lower()]

    already_matched = {m.lower() for m in matches}
    matches.update(_fuzzy_matches(query, already_matched))
    return matches


def relevant_drug_names(query: str) -> set[str]:
    """Canonical corpus drug names the query is actually asking about — both directly
    named ("warfarin") and alias-resolved ("paracetamol" -> acetaminophen). Used by
    routes_ask.py to keep a passage that merely *mentions* one of these drugs inside
    some unrelated drug's own interactions section (e.g. Azithromycin's label happens
    to have its own "7.2 Warfarin" section) from outranking that drug's own document,
    which is the authoritative source for describing its side of the interaction.
    """
    lowered = query.lower()
    direct = {g for g in CORPUS_GENERICS if re.search(rf"\b{re.escape(g)}\b", lowered)}
    aliased = {c for canonicals in find_aliases(query).values() for c in canonicals if c in CORPUS_GENERICS}
    return direct | aliased


def expand_query(query: str) -> str:
    """Appends canonical generic-name terms for any recognized alias found in the query.
    Retrieval-only — never shown to the user or passed to the LLM as the question asked.
    """
    aliases = find_aliases(query)
    if not aliases:
        return query

    matched_terms = [canonical for canonicals in aliases.values() for canonical in canonicals]
    unique_terms = dict.fromkeys(matched_terms)  # de-dup, preserve first-seen order
    return f"{query} {' '.join(unique_terms)}"
