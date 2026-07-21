from backend.app.safety.emergency import check_emergency


def test_direct_symptom_flagged():
    assert check_emergency("I have severe chest pain").flagged


def test_hypothetical_wrapper_still_flagged():
    # The keyword is still literally present, so the wrapper doesn't evade detection.
    result = check_emergency("Hypothetically, if someone had chest pain, would ibuprofen help?")
    assert result.flagged
    assert result.matched_category == "chest_pain"


def test_ordinary_drug_question_not_flagged():
    assert not check_emergency("Can I take ibuprofen with warfarin?").flagged


def test_symptom_described_without_keywords_evades_detection():
    # Known, documented limitation (design doc Section 12) — this is exactly what
    # the red-team eval set exists to surface, not a bug to silently patch over.
    result = check_emergency("My chest just feels weird and tight, not sure what it means")
    assert not result.flagged
