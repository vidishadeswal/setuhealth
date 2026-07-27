from backend.app.generation.prompt import build_prompt
from backend.app.retrieval.reranker import RerankedChunk

CHUNK = RerankedChunk(
    chunk_id="c1",
    content="Chronic oral acetaminophen use may increase INR in patients on warfarin.",
    page_number=3,
    document_title="Acetaminophen Interactions",
    rerank_score=1.0,
)


def test_no_alias_hints_omits_hint_block():
    prompt = build_prompt("Can I take acetaminophen with warfarin?", [CHUNK])
    assert "Known mappings" not in prompt


def test_alias_hints_are_surfaced_to_the_model():
    # Regression test: retrieval can find the right passage for a brand-name query
    # ("Dolo" -> acetaminophen) via query_expansion, but the model only ever sees the
    # user's original phrasing — without this hint it has no way to connect "Dolo" in
    # the question to "acetaminophen" in the passage, and will (accurately, from its
    # perspective) say the passages don't address the question.
    prompt = build_prompt(
        "Can I take Dolo with warfarin?", [CHUNK], alias_hints={"Dolo": ["acetaminophen", "paracetamol"]}
    )
    assert "Known mappings" in prompt
    assert '"Dolo" refers to: acetaminophen, paracetamol' in prompt
