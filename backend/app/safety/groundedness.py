"""Post-generation groundedness check (design doc Section 11). An instruction to "only
use the passages" is a request, not a guarantee — this checks the actual output. Each
sentence of the generated answer is compared by embedding similarity against the
retrieved chunks; anything below threshold is flagged as an unsupported claim. This is
also what produces the hallucination-rate number in the eval report — not a claim, a
measurement.

Per-sentence best-match chunk indices double as the source of truth for citations
(api/routes_ask.py): deriving "what was this answer actually grounded in" from the same
embedding evidence used to check groundedness is more robust than parsing the LLM's own
citation markers out of its prose, which a small local model won't always format exactly
as instructed, and which would otherwise go missing if the one sentence carrying the
marker got stripped as ungrounded.
"""

import re
from dataclasses import dataclass

import numpy as np

from backend.app.retrieval.embeddings import embed_passages

GROUNDEDNESS_THRESHOLD = 0.55
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


@dataclass
class SentenceGrounding:
    sentence: str
    is_grounded: bool
    best_chunk_index: int | None
    similarity: float


@dataclass
class GroundednessResult:
    sentences: list[SentenceGrounding]

    @property
    def grounded_sentences(self) -> list[SentenceGrounding]:
        return [s for s in self.sentences if s.is_grounded]

    @property
    def ungrounded_sentences(self) -> list[str]:
        return [s.sentence for s in self.sentences if not s.is_grounded]

    @property
    def total_sentences(self) -> int:
        return len(self.sentences)

    @property
    def ungrounded_count(self) -> int:
        return len(self.ungrounded_sentences)

    @property
    def is_fully_grounded(self) -> bool:
        return self.ungrounded_count == 0


def check_groundedness(answer_text: str, source_chunks: list[str]) -> GroundednessResult:
    sentences = split_sentences(answer_text)
    if not sentences or not source_chunks:
        return GroundednessResult(
            sentences=[SentenceGrounding(s, is_grounded=False, best_chunk_index=None, similarity=0.0) for s in sentences]
        )

    sentence_vectors = embed_passages(sentences)
    chunk_vectors = embed_passages(source_chunks)

    # Vectors are normalized, so dot product is cosine similarity.
    similarity_matrix = sentence_vectors @ chunk_vectors.T
    best_chunk_per_sentence = np.argmax(similarity_matrix, axis=1)
    best_similarity_per_sentence = np.max(similarity_matrix, axis=1)

    graded = [
        SentenceGrounding(
            sentence=sentence,
            is_grounded=bool(similarity >= GROUNDEDNESS_THRESHOLD),
            best_chunk_index=int(best_idx),
            similarity=float(similarity),
        )
        for sentence, best_idx, similarity in zip(sentences, best_chunk_per_sentence, best_similarity_per_sentence)
    ]
    return GroundednessResult(sentences=graded)
