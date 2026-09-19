from scripts.build_corpus_from_fda import split_pages


def test_decimals_in_pk_text_are_not_treated_as_section_headers():
    # Regression: cyclosporine's label contained "3.49 L/kg" and "4.3 Age 8 to 15",
    # which an unrestricted digit.digit regex split into meaningless fragment pages.
    text = "Clearance was 3.49 L/kg and 0.369 L/hr/kg. 4.3 Age 8 to 15 values were similar."
    assert len(split_pages(text)) == 1


def test_real_section_7_headers_still_split():
    text = "7.1 First Topic Some text here. 7.2 Second Topic More text here."
    pages = split_pages(text)
    assert len(pages) == 2
    assert "7.1" in pages[0][0] and "7.2" in pages[1][0]


def test_cross_references_inside_numbered_sections_do_not_fragment_the_section():
    # Regression: tamsulosin's 7.1 contains "...Clinical Pharmacology (12.3)]. The effects of..."
    # repeatedly; the topic-header fallback treated each as a new page.
    text = (
        "7.1 Cytochrome P450 Interactions Effects one [see Clinical Pharmacology (12.3)]. "
        "The effects of a moderate inhibitor were studied [see Clinical Pharmacology (12.3)]. "
        "Concomitant treatment with paroxetine increased exposure. 7.2 Other Agents Text here."
    )
    pages = split_pages(text)
    assert len(pages) == 2
