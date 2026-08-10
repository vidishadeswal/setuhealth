from backend.app.ingestion.pipeline import _grounded_chunk_content


def test_drug_name_is_prepended_to_content_pages():
    # Regression test: furosemide's Lithium interaction page never says "furosemide"
    # at all (the source text only says "diuretics"), which made a cross-encoder rank
    # it dead last against a query naming furosemide, behind unrelated passages that
    # happened to repeat the drug's name. Prepending the drug name fixes this for any
    # chunk whose source prose relies on document-level context it no longer has once
    # split into its own page.
    result = _grounded_chunk_content(
        "Lithium generally should not be given with diuretics.", page_number=5, drug_name="Furosemide"
    )
    assert result.startswith("Furosemide:")
    assert "Lithium generally should not be given with diuretics." in result


def test_attribution_header_page_is_left_unchanged():
    # Page 1 is the SetuHealth attribution/disclaimer header, not interaction content
    # — it already names the drug prominently and doesn't need the prefix.
    header = "SetuHealth Source Document — FDA-Approved Drug Labeling\n\nGeneric name: FUROSEMIDE"
    assert _grounded_chunk_content(header, page_number=1, drug_name="Furosemide") == header
