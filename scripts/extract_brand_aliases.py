"""One-off script: extracts brand-name -> generic-name aliases for the drugs already in
the SetuHealth corpus, from a public Kaggle medicine dataset (Medicine_Details.csv,
sourced from 1mg.com listings — unverified provenance, used only to widen retrieval
query matching, never as cited/grounding content).

Filters applied, deliberately conservative:
  - Only rows whose Composition mentions one of the corpus's 6 covered generics
  - Single-ingredient formulations only (no "+" combination products) — a combination
    product's brand name doesn't unambiguously mean "this one active ingredient"
  - Oral/systemic forms only (excludes eye drops, injections, ointments, etc.) — the
    corpus's interaction content is about oral use
  - Brand name reduced to its root (strips dosage strength and dosage-form words)

Not run automatically as part of ingestion — run once, review the output, commit the
resulting JSON. Re-run if the corpus's covered generics change.

Usage: python scripts/extract_brand_aliases.py <path-to-Medicine_Details.csv>
"""

import csv
import json
import re
import sys
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "backend" / "app" / "retrieval" / "data" / "brand_aliases.json"

# generic name as it appears in Composition -> canonical name as used in the corpus.
# The Kaggle dataset is India-sourced (1mg.com), so a few entries use British/Indian
# spelling conventions that differ from the US FDA generic name used in the corpus.
CORPUS_GENERICS = {
    "warfarin": "warfarin",
    "doxycycline": "doxycycline",
    "ciprofloxacin": "ciprofloxacin",
    "sertraline": "sertraline",
    "simvastatin": "simvastatin",
    "paracetamol": "acetaminophen",
    "ibuprofen": "ibuprofen",
    "clopidogrel": "clopidogrel",
    "digoxin": "digoxin",
    "methotrexate": "methotrexate",
    "phenytoin": "phenytoin",
    "ciclosporin": "cyclosporine",  # British spelling in this dataset
    "metformin": "metformin",
    "atorvastatin": "atorvastatin",
    "lisinopril": "lisinopril",
    "omeprazole": "omeprazole",
    "thyroxine": "levothyroxine",  # dataset uses "thyroxine", not "levothyroxine"
    "amoxycillin": "amoxicillin",  # British spelling in this dataset
    "azithromycin": "azithromycin",
    "fluoxetine": "fluoxetine",
    "amlodipine": "amlodipine",
    "losartan": "losartan",
    "hydrochlorothiazide": "hydrochlorothiazide",
    "gabapentin": "gabapentin",
    "citalopram": "citalopram",
    "furosemide": "furosemide",
    "montelukast": "montelukast",
    "pantoprazole": "pantoprazole",
    "metoprolol": "metoprolol",
    "spironolactone": "spironolactone",
    "allopurinol": "allopurinol",
    "duloxetine": "duloxetine",
    "escitalopram": "escitalopram",
    "tamsulosin": "tamsulosin",
    # No Kaggle matches found for tramadol, prednisone, or alprazolam under any spelling
    # checked — covered by the hand-curated brand aliases in query_expansion.py
    # (DRUG_ALIASES) instead. Note: the dataset's "prednisolone" entries are a
    # different, related-but-distinct corticosteroid, not an alternate spelling of
    # prednisone — not aliased here to avoid conflating the two drugs.
}

NON_ORAL_RE = re.compile(r"\b(eye|ear|ointment|injection|infusion|drops?|cream|gel|lotion|inhaler|nasal|topical)\b", re.IGNORECASE)
FORM_WORD_RE = re.compile(r"\b(tablet|capsule|syrup|suspension|oral|drops?|injection|infusion|ointment)\b.*$", re.IGNORECASE)
TRAILING_NUM_RE = re.compile(r"\s*[\d.]+\s*(mg|mcg|ml|gm|g|%|iu)?\s*$", re.IGNORECASE)


def extract_root(name: str) -> str:
    root = FORM_WORD_RE.sub("", name).strip()
    prev = None
    while prev != root:
        prev = root
        root = TRAILING_NUM_RE.sub("", root).strip()
    return root.strip(" /-.").strip()


def main(csv_path: str) -> None:
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    aliases: dict[str, list[str]] = {}
    for generic, canonical in CORPUS_GENERICS.items():
        matches = [
            r for r in rows
            if generic in r["Composition"].lower()
            and "+" not in r["Composition"]
            and not NON_ORAL_RE.search(r["Medicine Name"])
        ]
        roots = sorted({extract_root(m["Medicine Name"]) for m in matches} - {""})
        for root in roots:
            key = root.lower()
            aliases.setdefault(key, [])
            if canonical not in aliases[key]:
                aliases[key].append(canonical)

    output = {
        "_source": "Kaggle 'Medicine Dataset' (Medicine_Details.csv, 1mg.com listings) — unverified provenance",
        "_filters": "single-ingredient oral formulations only, matched against the corpus's covered generics",
        "_usage": "retrieval-query expansion only — never used as cited/grounding content",
        "_count": len(aliases),
        "aliases": aliases,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(aliases)} brand aliases to {OUTPUT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/extract_brand_aliases.py <path-to-Medicine_Details.csv>")
        sys.exit(1)
    main(sys.argv[1])
