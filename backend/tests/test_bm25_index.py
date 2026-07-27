from backend.app.retrieval.bm25_index import tokenize


def test_hyphenated_compound_matches_unhyphenated_query_term():
    # Regression test: corpus text says "azole anti-fungals" (FDA label style), a real
    # query says "antifungal" — plain regex tokenization on "-" as a boundary makes
    # these ["anti", "fungals"] vs ["antifungal"], sharing zero tokens, which was a real
    # contributor to a near-miss confidence score (43.8% vs a 45% threshold) in testing.
    assert "antifungals" in tokenize("azole anti-fungals")
    assert "antifungal" in tokenize("Can I take an antifungal?")


def test_other_hyphenated_corpus_terms_join_consistently():
    assert tokenize("anti-coagulant") == ["anticoagulant"]
    assert tokenize("co-administration") == ["coadministration"]
