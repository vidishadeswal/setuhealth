"""Restores real line breaks in a markdown table embedded in chunk content before it
goes into the LLM prompt.

Chunk content is stored with all whitespace (including the source document's original
newlines) collapsed to single spaces — see ingestion/chunker.py's word-join. A
flattened markdown table still parses fine for a human or for the frontend's own
row-boundary recovery (frontend/src/utils/parsePassage.ts), but a small local model
reading one dense unbroken line of "| Amiodarone | 70% | NA | | Captopril | 58% | 39%
| ..." measurably misattributes values between adjacent rows — confirmed live: asked
"does amiodarone increase digoxin concentrations", it answered "150%", which is
Dronedarone's AUC value three rows down, not Amiodarone's real 70%. Groundedness
checking doesn't catch this because 150% genuinely appears verbatim in the source
passage — it's just attached to the wrong drug.

This reconstructs actual newlines and pipe alignment for exactly the substring the
frontend already knows how to parse (see parsePassageContent for the matching row-
boundary logic: an empty split segment between adjacent "|" marks a row break), so the
model reads one row per line instead of one continuous run.
"""

def reconstruct_tables(content: str) -> str:
    first_pipe = content.find("|")
    last_pipe = content.rfind("|")
    if first_pipe == -1 or last_pipe == first_pipe:
        return content

    prefix = content[:first_pipe].strip()
    suffix = content[last_pipe + 1 :].strip()
    table_region = content[first_pipe : last_pipe + 1]

    raw_cells = [c.strip() for c in table_region.split("|")]
    raw_cells = raw_cells[1:-1]  # drop the empty edges from the outer pipes

    rows: list[list[str]] = [[]]
    for cell in raw_cells:
        if cell == "":
            rows.append([])
        else:
            rows[-1].append(cell)
    rows = [r for r in rows if r]

    if len(rows) < 2:
        return content

    table_markdown = "\n".join(f"| {' | '.join(row)} |" for row in rows)

    parts = [p for p in (prefix, table_markdown, suffix) if p]
    return "\n".join(parts)
