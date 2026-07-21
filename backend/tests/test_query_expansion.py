from backend.app.retrieval.query_expansion import expand_query


def test_brand_name_expands_to_generic():
    expanded = expand_query("Can I take Advil with warfarin?")
    assert "ibuprofen" in expanded.lower()
    assert expanded.startswith("Can I take Advil with warfarin?")


def test_colloquial_phrase_expands():
    expanded = expand_query("Can I take blood thinner with Advil?")
    assert "warfarin" in expanded.lower()
    assert "ibuprofen" in expanded.lower()


def test_indian_brand_name_expands():
    expanded = expand_query("can i take dolo with warfarin")
    assert "acetaminophen" in expanded.lower()
    assert "paracetamol" in expanded.lower()


def test_no_alias_leaves_query_unchanged():
    query = "What is the interaction between simvastatin and grapefruit juice?"
    assert expand_query(query) == query


def test_does_not_match_substring_inside_another_word():
    # "cipro" must not fire on "ciprofloxacin" itself and produce a duplicated/odd expansion
    expanded = expand_query("Is ciprofloxacin safe with calcium?")
    assert expanded == "Is ciprofloxacin safe with calcium?"
