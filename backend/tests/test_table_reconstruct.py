from backend.app.generation.table_reconstruct import reconstruct_tables

FLATTENED_TABLE = (
    "Digoxin concentrations increased greater than 50%: "
    "| Drug | Digoxin Serum Concentration Increase | Digoxin AUC Increase | "
    "|---|---|---| "
    "| Amiodarone | 70% | NA | "
    "| Captopril | 58% | 39% | "
    "| Dronedarone | NA | 150% | "
    "Recommendation: monitor digoxin levels."
)


def test_reconstructs_one_row_per_line():
    result = reconstruct_tables(FLATTENED_TABLE)
    lines = result.split("\n")
    amiodarone_line = next(line for line in lines if "Amiodarone" in line)
    dronedarone_line = next(line for line in lines if "Dronedarone" in line)
    # The regression this guards against: a flattened single-line table let the model
    # attach a neighboring row's value ("150%", Dronedarone's AUC) to Amiodarone.
    # Reconstructed, each drug's real values must live on its own line, not bleed
    # into an adjacent row.
    assert "70%" in amiodarone_line
    assert "150%" not in amiodarone_line
    assert "150%" in dronedarone_line


def test_preserves_prefix_and_suffix_prose():
    result = reconstruct_tables(FLATTENED_TABLE)
    assert result.startswith("Digoxin concentrations increased greater than 50%:")
    assert result.endswith("Recommendation: monitor digoxin levels.")


def test_plain_prose_without_a_table_is_returned_unchanged():
    prose = "Chronic oral acetaminophen use may increase INR in patients on warfarin."
    assert reconstruct_tables(prose) == prose


def test_single_stray_pipe_is_not_treated_as_a_table():
    text = "The ratio is roughly 2 | 1 in most patients."
    assert reconstruct_tables(text) == text
