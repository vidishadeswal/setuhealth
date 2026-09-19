"""Reusable corpus-builder: converts raw openFDA drug-label JSON (one file per drug,
each the single result object from api.fda.gov/drug/label.json) into a SetuHealth
corpus .txt file using the same ===PAGE N=== format as the original hand-built
documents (corpus/sample/warfarin_interactions.txt etc).

Page boundaries are placed at natural FDA structure — numbered subsections ("7.1 ",
"7.2 ", ...) where present, else "Table N:" markers, else the whole section becomes
one page. This script only decides citation-page boundaries; the actual dense,
token-sized chunking within a page still happens at ingestion time via
ingestion/chunker.py, exactly as it already does for the original 6 documents (whose
largest pages are themselves auto-split into several retrieval chunks under one page
label). Hand-crafting per-drug boundaries the way the original 6 were built isn't
practical at this scale, so this trades a little citation granularity for coverage.

Usage: python scripts/build_corpus_from_fda.py <dir-of-openfda-json> <output-dir>
Input JSON files must be named <drug_slug>.json and contain a "drug_interactions" key
(a single openFDA label result object, not the full API response envelope).
"""

import json
import re
import sys
from pathlib import Path

# Real FDA subsection headers ("7.1 P-Glycoprotein...") are followed directly by a
# capitalized word. Bracketed cross-references to the same numbers ("( 7.1 )", "( 7.2 ,
# 7.3 , 12.3 )") are followed by a comma or closing paren instead, so this lookahead
# naturally excludes them without needing a lookbehind for the opening paren.
# Restricted to section 7 (the FDA "Drug Interactions" section itself): an unrestricted
# \d.\d+ also matches decimals inside pharmacokinetic text ("3.49 L/kg", "4.3 Age 8..."),
# which turned one cyclosporine label into nine meaningless one-line "pages".
SECTION_RE = re.compile(r"(?<![\d.])7\.\d+(?=\s+[A-Z])")
TABLE_RE = re.compile(r"Table \d+:")
BOILERPLATE_LEAD_RE = re.compile(r"^\d+\s+DRUG INTERACTIONS\s*")

# Secondary split for "Table N:"-fallback pages: many such labels list one interacting
# drug class per subsection, each written as "<Class Name> Clinical Impact: <text>
# Intervention: <text>" with no other markup — reliable because "Clinical Impact:" is a
# consistent literal FDA phrase, unlike the free-form class-name headers themselves.
# Without this, a page like ibuprofen's or tramadol's ends up as one multi-topic blob
# (ACE-inhibitors, aspirin, lithium, warfarin, etc. all bundled together), which in
# testing caused the LLM to occasionally answer from the wrong drug class in that blob
# even though the right one was right there in the same chunk.
_HEADER_WORD = r"(?:[A-Z(][\w()/\-]*|of|and|the|or)"
# Colon after "Clinical Impact" is optional — some labels (e.g. allopurinol) write
# "Clinical Impact" and "Intervention" as bare bolded-then-stripped words with no
# punctuation. Matching this format here, not just the colon form, keeps it from
# falling through to the topic-header fallback below, which would otherwise treat the
# bare word "Intervention" as its own topic header and split a Clinical
# Impact/Intervention pair — which belongs on one page — into two.
CLINICAL_IMPACT_RE = re.compile(rf"((?:{_HEADER_WORD}\s+){{0,10}}{_HEADER_WORD})\s+Clinical Impact:?")
TABLE_PREAMBLE_RE = re.compile(
    r"^Table \d+:\s*Clinically (?:Significant|Relevant) Drug Interactions? (?:with|Affecting)[^.]*?"
    r"(?=\b[A-Z][a-zA-Z0-9()/\-]*(?:\s+(?:of|and|the|or)\s+[A-Z(][\w()/\-]*)* Clinical Impact:?\s)"
)


def _split_on_clinical_impact(title: str, body: str, allow_topic_fallback: bool = True) -> list[tuple[str, str]]:
    cleaned = TABLE_PREAMBLE_RE.sub("", body)
    matches = list(CLINICAL_IMPACT_RE.finditer(cleaned))
    if len(matches) < 2:
        # The loose topic-header guess is only safe on unstructured text. Inside an
        # already-numbered section it misfires on cross-references ("...and Clinical
        # Pharmacology (12.3)]. The effects of...") and fragments the section (tamsulosin
        # 7.1 became eight pages titled "— and Clinical and Clinical Pharmacology").
        return _split_on_topic_headers(title, body) if allow_topic_fallback else [(title, body)]

    pages = []
    for i, m in enumerate(matches):
        start = m.start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(cleaned)
        section_body = cleaned[start:end].strip()
        if section_body:
            pages.append((f"{title} — {m.group(1).strip()}", section_body))
    return pages


# Third-tier fallback for labels with neither numbered subsections, "Table N:", nor
# "Clinical Impact:" — some FDA labels instead list one interacting drug/class per
# short noun-phrase header directly followed by its effect sentence, with no
# announcing marker at all (e.g. prednisone: "...Antibiotics Macrolide antibiotics
# have been reported to cause..."). Detected by two title-case word-runs appearing
# back to back right after a sentence boundary — ordinary prose only ever has one
# capitalized run per sentence start, so a second one immediately after is a strong
# signal of [header][new sentence] rather than a single continuing sentence. Not every
# label has this structure (some, like furosemide's, are genuinely unstructured
# narrative prose with no per-topic boundaries at all) — those correctly get 0 matches
# and stay as one page, which chunk_page() still splits reasonably by sentence count.
_TOPIC_HEADER_WORD = r"(?:[A-Z][\w()/,\-]*|and|or|the|including|with|without|of)"
TOPIC_HEADER_RE = re.compile(
    rf"(?:(?<=[.\)])\s+|^)((?:{_TOPIC_HEADER_WORD}\s+){{0,5}}{_TOPIC_HEADER_WORD})\s+(?=[A-Z][a-z])"
)


# Structural words that can look like a topic header (capitalized, sentence-initial)
# but aren't actually naming an interacting drug or class — matching one here would
# split a paragraph away from the very topic header it belongs under.
_NON_TOPIC_HEADERS = {"intervention", "examples", "clinical impact"}


def _split_on_topic_headers(title: str, body: str) -> list[tuple[str, str]]:
    matches = [m for m in TOPIC_HEADER_RE.finditer(body) if m.group(1).strip().lower() not in _NON_TOPIC_HEADERS]
    if len(matches) < 2:
        return [(title, body)]

    pages = []
    for i, m in enumerate(matches):
        start = m.start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(body)
        section_body = body[start:end].strip()
        if section_body:
            pages.append((f"{title} — {m.group(1).strip()}", section_body))
    return pages

HEADER = """SetuHealth Source Document — FDA-Approved Drug Labeling

Source: openFDA (api.fda.gov), U.S. Food and Drug Administration.
This text is FDA-approved prescribing information, a work of the U.S. federal
government and not subject to copyright protection in the United States
(17 U.S.C. § 105). Retrieved programmatically via the openFDA drug label API.

Generic name: {generic}
Brand name(s): {brand}
Label set_id: {set_id}
Label effective date: {effective_date}

This is prescribing information written for healthcare professionals, reproduced
here to demonstrate a retrieval system — it is not a substitute for a pharmacist
or prescriber, and this tool remains a PROTOTYPE, not for real clinical decisions.
"""


def split_pages(text: str) -> list[tuple[str, str]]:
    text = BOILERPLATE_LEAD_RE.sub("", text).strip()

    section_marks = [(m.start(), m.group()) for m in SECTION_RE.finditer(text)]
    table_marks = [(m.start(), m.group().rstrip(":")) for m in TABLE_RE.finditer(text)]
    marks = section_marks or table_marks

    if not marks:
        return _split_on_clinical_impact("Drug Interactions", text)

    pages = []
    for i, (start, label) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[start:end].strip()
        if body:
            pages.extend(_split_on_clinical_impact(f"Section {label} — Drug Interactions", body, allow_topic_fallback=False))
    return pages


def format_effective_date(raw: str) -> str:
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw or "unknown"


def build_document(record: dict) -> str:
    openfda = record.get("openfda", {})
    generic = ", ".join(openfda.get("generic_name", ["UNKNOWN"]))
    brand = ", ".join(openfda.get("brand_name", [])) or generic
    set_id = record.get("set_id", "unknown")
    effective_date = format_effective_date(record.get("effective_time", ""))

    header = HEADER.format(generic=generic, brand=brand, set_id=set_id, effective_date=effective_date)

    text = record["drug_interactions"][0]
    pages = split_pages(text)

    out = [f"===PAGE 1===\n{header}"]
    for i, (title, body) in enumerate(pages, start=2):
        out.append(f"===PAGE {i}===\n{title}\n\n{body}")
    return "\n\n".join(out) + "\n"


def main(json_dir: str, output_dir: str) -> None:
    json_path = Path(json_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for f in sorted(json_path.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            record = json.load(fh)
        if "drug_interactions" not in record:
            print(f"skip {f.name}: no drug_interactions field")
            continue
        drug_slug = f.stem
        doc = build_document(record)
        out_file = out_path / f"{drug_slug}_interactions.txt"
        out_file.write_text(doc, encoding="utf-8")
        page_count = doc.count("===PAGE")
        print(f"wrote {out_file.name} ({len(doc)} chars, {page_count} pages)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/build_corpus_from_fda.py <dir-of-openfda-json> <output-dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
