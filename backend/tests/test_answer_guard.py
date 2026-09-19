from backend.app.generation.answer_guard import enforce_severity_wording
from backend.app.generation.prompt import SYSTEM_INSTRUCTION


def test_contraindicated_is_softened_when_no_passage_says_it():
    passages = ["Lithium generally should not be given with diuretics."]
    out = enforce_severity_wording("This combination is contraindicated.", passages)
    assert "contraindicated" not in out
    assert "not recommended" in out


def test_contraindicated_is_kept_when_a_passage_uses_the_word():
    passages = ["Concomitant use with strong CYP3A4 inhibitors is contraindicated."]
    answer = "This combination is contraindicated."
    assert enforce_severity_wording(answer, passages) == answer


def test_prompt_has_no_placeholder_x_for_the_model_to_copy():
    # Regression: the prompt's own example "increases the risk of X" leaked into a
    # live answer verbatim ("Grapefruit juice increases the risk of X").
    assert "risk of X" not in SYSTEM_INSTRUCTION
    assert "monitor for X" not in SYSTEM_INSTRUCTION


def test_blanket_denial_dropped_when_an_interaction_was_already_stated():
    from backend.app.generation.answer_guard import drop_self_contradiction

    out = drop_self_contradiction(
        "Sertraline and tramadol increase the risk of serotonin syndrome. "
        "No interaction between these two is noted in the passages."
    )
    assert "No interaction" not in out
    assert "serotonin syndrome" in out


def test_lone_denial_is_kept():
    from backend.app.generation.answer_guard import drop_self_contradiction

    text = "No interaction between these two is noted in the passages."
    assert drop_self_contradiction(text) == text
