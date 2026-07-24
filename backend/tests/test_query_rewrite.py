from backend.app.generation.query_rewrite import _parse_variants, has_lexical_overlap


def test_parses_well_formed_response():
    raw = (
        "REWRITE: Can I take a blood thinner with ibuprofen?\n"
        "REWRITE: Does warfarin interact with NSAIDs?\n"
        "HYDE: Concomitant use of warfarin and NSAIDs increases bleeding risk."
    )
    variants = _parse_variants(raw)
    assert variants == [
        "Can I take a blood thinner with ibuprofen?",
        "Does warfarin interact with NSAIDs?",
        "Concomitant use of warfarin and NSAIDs increases bleeding risk.",
    ]


def test_ignores_commentary_and_blank_lines():
    raw = (
        "Sure, here are the variants:\n\n"
        "REWRITE: Can I take Advil with warfarin?\n"
        "\n"
        "HYDE: Ibuprofen may increase warfarin's anticoagulant effect.\n"
        "Let me know if you need anything else."
    )
    variants = _parse_variants(raw)
    assert variants == [
        "Can I take Advil with warfarin?",
        "Ibuprofen may increase warfarin's anticoagulant effect.",
    ]


def test_empty_or_garbage_response_returns_empty_list():
    assert _parse_variants("") == []
    assert _parse_variants("I'm not sure how to help with that.") == []


def test_none_response_for_out_of_scope_question_returns_empty_list():
    # The model is instructed to respond NONE rather than fabricate drug-related
    # content for a question that isn't about drug interactions at all — this must
    # never produce fake variants that could inflate confidence on an unrelated query.
    assert _parse_variants("NONE") == []


def test_lexical_overlap_true_when_real_query_word_appears_in_chunk():
    query = "Can I take blood thinner with Advil? warfarin anticoagulant ibuprofen"
    chunk = "Non-steroidal Anti-Inflammatory Agents including ibuprofen increase bleeding risk with warfarin."
    assert has_lexical_overlap(query, chunk)


def test_lexical_overlap_false_for_genuinely_unrelated_query():
    # Regression test: a fabricated HyDE sentence about "geography-related medications"
    # once scored this exact chunk highly for this exact query — the overlap gate is
    # what must catch it now.
    query = "What is the capital of France?"
    chunk = "Serotonin syndrome has been reported with SNRIs and SSRIs, particularly with concomitant use of other serotonergic drugs."
    assert not has_lexical_overlap(query, chunk)


def test_lexical_overlap_ignores_stopwords_only_matches():
    query = "What is the interaction with this?"
    chunk = "The patient should take this with food."
    assert not has_lexical_overlap(query, chunk)
