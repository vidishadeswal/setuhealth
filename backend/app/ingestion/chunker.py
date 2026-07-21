"""Paragraph-bounded chunking (design doc Section 07): split on paragraph boundaries,
cap chunk size in tokens, small overlap so a chunk never hard-cuts mid-thought at a
table row or list item. Token count is approximated by whitespace word count, which is
adequate at this corpus size and avoids pulling in a tokenizer just for chunk sizing.
"""

from dataclasses import dataclass


@dataclass
class ChunkResult:
    content: str
    page_number: int


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_page(text: str, page_number: int, token_size: int, overlap_ratio: float) -> list[ChunkResult]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    overlap_words = max(1, int(token_size * overlap_ratio))
    chunks: list[ChunkResult] = []
    buffer_words: list[str] = []

    def flush():
        if buffer_words:
            chunks.append(ChunkResult(content=" ".join(buffer_words).strip(), page_number=page_number))

    for paragraph in paragraphs:
        para_words = paragraph.split()

        # A single paragraph longer than the cap is split on its own — still whole
        # sentences where possible, since a mid-sentence cut is worse than a slightly
        # oversized chunk.
        if len(para_words) > token_size:
            for sentence in _split_long_paragraph(paragraph, token_size):
                buffer_words.extend(sentence.split())
                if len(buffer_words) >= token_size:
                    flush()
                    buffer_words = buffer_words[-overlap_words:]
            continue

        if len(buffer_words) + len(para_words) > token_size and buffer_words:
            flush()
            buffer_words = buffer_words[-overlap_words:]

        buffer_words.extend(para_words)

    flush()
    return chunks


def _split_long_paragraph(paragraph: str, token_size: int) -> list[str]:
    sentences = [s.strip() for s in paragraph.replace("\n", " ").split(". ") if s.strip()]
    out, buf = [], []
    for s in sentences:
        buf.append(s if s.endswith(".") else s + ".")
        if _word_count(" ".join(buf)) >= token_size:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out or [paragraph]
