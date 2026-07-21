"""Plain-text extraction only — no OCR. Scanned/image-only PDFs are out of scope for this
prototype (see the design doc, Section 02): real ingestion effort with no retrieval-quality payoff.
"""

from pathlib import Path

from pypdf import PdfReader

PAGE_MARKER = "===PAGE"


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Returns a list of (page_number, page_text), 1-indexed."""
    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path)
    if path.suffix.lower() == ".txt":
        return _extract_txt(path)
    raise ValueError(f"Unsupported source type: {path.suffix} (plain text extraction only)")


def _extract_pdf(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def _extract_txt(path: Path) -> list[tuple[int, str]]:
    """Sample .txt corpus files mark page breaks with a line like `===PAGE 3===`."""
    raw = path.read_text(encoding="utf-8")
    pages: list[tuple[int, str]] = []
    current_page = 1
    buffer: list[str] = []

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(PAGE_MARKER):
            if buffer:
                pages.append((current_page, "\n".join(buffer).strip()))
                buffer = []
            digits = "".join(c for c in stripped if c.isdigit())
            current_page = int(digits) if digits else current_page + 1
        else:
            buffer.append(line)

    if buffer:
        pages.append((current_page, "\n".join(buffer).strip()))

    return [(p, t) for p, t in pages if t]
