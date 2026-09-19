import numpy as np

from backend.app.safety.groundedness import pick_citation_index


def test_picks_highest_similarity_when_no_preference():
    assert pick_citation_index(np.array([0.6, 0.8, 0.7]), None) == 1


def test_prefers_asked_about_drug_when_nearly_tied():
    # Sertraline (index 0) vs Fluoxetine (index 1) share near-identical boilerplate; the
    # user asked about sertraline, so a 0.02 gap must not credit the sibling drug.
    assert pick_citation_index(np.array([0.78, 0.80]), {0}) == 0


def test_does_not_prefer_a_clearly_worse_match():
    assert pick_citation_index(np.array([0.50, 0.85]), {0}) == 1


def test_best_already_preferred_is_kept():
    assert pick_citation_index(np.array([0.9, 0.7]), {0}) == 0
