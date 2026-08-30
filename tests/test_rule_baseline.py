from src.eval.rule_baseline import ranked_labels, rule_choice

THR = {"symmetry_strong": 0.7, "balance_skew": 0.25,
       "edge_near": 0.05, "center_align": 0.08}

OPTS = {"A": "Center", "B": "Symmetrical", "C": "Left heavy", "D": "Short side"}


def _features(sym_h=0.3, balance=0.0, min_edge=0.3, d_center=0.3, subject=True):
    f = {
        "symmetry": {"sym_h": sym_h, "sym_v": 0.2},
        "balance": {"balance": balance},
        "no_subject": not subject,
        "subject": ({"d_center": d_center,
                     "edge": {"min_edge_dist": min_edge}} if subject else None),
    }
    return f


def test_symmetry_wins_first():
    f = _features(sym_h=0.9, balance=-0.5)
    assert ranked_labels(f, THR)[0] == "Symmetrical"
    assert rule_choice(f, OPTS, THR) == "B"


def test_left_heavy_from_negative_balance():
    f = _features(balance=-0.4)
    assert rule_choice(f, OPTS, THR) == "C"


def test_short_side_and_center_rules():
    assert rule_choice(_features(min_edge=0.02), OPTS, THR) == "D"
    assert rule_choice(_features(d_center=0.03), OPTS, THR) == "A"


def test_default_balanced_falls_through_to_available_option():
    f = _features()  # nothing triggers -> Balanced, not in OPTS -> next ranked label
    letter = rule_choice(f, OPTS, THR)
    assert letter in OPTS


def test_combo_option_prefers_exact_single_label():
    opts = {"A": "Balanced, Symmetrical", "B": "Symmetrical", "C": "Center",
            "D": "Right heavy"}
    f = _features(sym_h=0.95)
    assert rule_choice(f, opts, THR) == "B"  # exact single-label beats combo


def test_combo_option_matched_when_no_exact():
    opts = {"A": "Balanced, Symmetrical", "B": "Center", "C": "Right heavy",
            "D": "Short side"}
    f = _features(sym_h=0.95)
    assert rule_choice(f, opts, THR) == "A"


def test_no_subject_skips_subject_rules():
    f = _features(subject=False, sym_h=0.2)
    labels = ranked_labels(f, THR)
    assert labels[0] == "Balanced"  # edge/center rules cannot fire
