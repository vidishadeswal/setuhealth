from backend.app.ingestion.chunker import chunk_page


def test_short_page_produces_one_chunk():
    text = "Warfarin interacts with NSAIDs.\n\nThe effect is additive."
    chunks = chunk_page(text, page_number=2, token_size=350, overlap_ratio=0.15)
    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_long_page_splits_into_multiple_chunks_with_overlap():
    token_size, overlap_ratio = 350, 0.15
    overlap_words = int(token_size * overlap_ratio)

    paragraph = " ".join(f"word{i}" for i in range(200))
    text = "\n\n".join([paragraph] * 4)  # ~800 words, well over the 350-word cap
    chunks = chunk_page(text, page_number=1, token_size=token_size, overlap_ratio=overlap_ratio)

    assert len(chunks) > 1
    first_words = chunks[0].content.split()
    second_words = chunks[1].content.split()
    # The tail of chunk 1 should reappear at the head of chunk 2 (the overlap window).
    assert first_words[-overlap_words:] == second_words[:overlap_words]


def test_empty_page_produces_no_chunks():
    assert chunk_page("   \n\n  ", page_number=1, token_size=350, overlap_ratio=0.15) == []
