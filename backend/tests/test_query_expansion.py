from backend.app.retrieval.query_expansion import expand_query, find_aliases, relevant_drug_names


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


def test_generated_indian_brand_names_are_loaded():
    # Regression test for the ~80 real brand names extracted from the Kaggle dataset
    # (scripts/extract_brand_aliases.py) — these are single-ingredient oral formulations
    # of the corpus's covered generics only, e.g. Warf (warfarin), Simvotin (simvastatin).
    assert "warfarin" in expand_query("Can I take Warf 5 with ibuprofen?").lower()
    assert "simvastatin" in expand_query("Is Simvotin safe with grapefruit juice?").lower()


def test_typo_of_generic_name_is_fuzzy_corrected():
    # Regression test: "can i take dolo with warfrin" scored 20% confidence in live
    # testing because "warfrin" never matched "warfarin" anywhere — a real, common
    # failure mode (people mistype drug names), not an edge case.
    aliases = find_aliases("can i take dolo with warfrin")
    assert "warfarin" in aliases["warfrin"]


def test_typo_of_brand_alias_is_fuzzy_corrected():
    aliases = find_aliases("is simvotan safe with grapefruit juice")
    assert "simvastatin" in aliases.get("simvotan", [])


def test_fuzzy_matching_does_not_false_positive_on_ordinary_words():
    for query in [
        "What are the side effects of taking multiple medications together?",
        "Should elderly patients avoid certain combinations?",
        "What foods should I avoid while taking antibiotics?",
    ]:
        assert find_aliases(query) == {}


def test_relevant_drug_names_finds_directly_named_and_alias_resolved_drugs():
    assert relevant_drug_names("Does paracetamol interact with warfarin?") == {"acetaminophen", "warfarin"}
    assert relevant_drug_names("Can I take blood thinner with Advil?") == {"warfarin", "clopidogrel", "ibuprofen"}


def test_relevant_drug_names_excludes_non_corpus_alias_terms():
    # "blood thinner" resolves to warfarin, clopidogrel, AND the generic term
    # "anticoagulant" — the last one isn't an actual corpus document and must be
    # dropped, not treated as a drug name to boost.
    assert "anticoagulant" not in relevant_drug_names("Can I take a blood thinner with food?")


def test_relevant_drug_names_empty_for_query_naming_no_corpus_drug():
    assert relevant_drug_names("What is the capital of France?") == set()


def test_category_level_alias_expands_to_every_corpus_drug_in_that_class():
    # Simvastatin and atorvastatin are both statins in the corpus, so "cholesterol
    # medicine" can't be pinned to one — it must expand to both.
    expanded = expand_query("Can I take my cholesterol medicine with an antifungal?").lower()
    assert "simvastatin" in expanded
    assert "atorvastatin" in expanded
    # Same for sertraline and fluoxetine as antidepressants.
    expanded = expand_query("Is my antidepressant safe with grapefruit?").lower()
    assert "sertraline" in expanded
    assert "fluoxetine" in expanded
