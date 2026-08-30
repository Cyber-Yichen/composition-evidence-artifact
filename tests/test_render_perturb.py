import pytest

from src.injection.perturb import perturb_features
from src.injection.prompts import build_prompt
from src.injection.render import (GROUPS, LexiconViolation, lexicon_guard,
                                  render_evidence)


def test_render_full(full_features, prompts):
    text = render_evidence(full_features, prompts)
    assert text.startswith("<composition_evidence>")
    assert "</composition_evidence>" in text
    assert "Subject:" in text and "Mirror similarity:" in text and "Depth:" in text
    assert "thirds-aligned" in text  # subject sits on the (1/3,1/3) intersection


def test_render_never_leaks_label_roots(full_features, prompts):
    """Word-root overlap with the SC answer-label set is forbidden."""
    text = render_evidence(full_features, prompts).lower()
    for root in ["center", "centre", "central", "symmetr", "balanc", "heavy",
                 "short side", "rule of thirds", "golden ratio"]:
        assert root not in text, root
    assert "centroid" in text and "midpoint" in text  # replacement wording present


def test_render_percentile_context(full_features, prompts):
    calib = {"sym_h": [0.1, 0.2, 0.3, 0.4, 0.5],
             "sym_v": [0.1, 0.2, 0.3, 0.4, 0.5],
             "abs_balance": [0.0, 0.1, 0.2, 0.3, 0.4]}
    text = render_evidence(full_features, prompts, calibration=calib)
    assert "percentile of film frames" in text
    plain = render_evidence(full_features, prompts)  # no calibration -> no context
    assert "percentile" not in plain
    numeric = render_evidence(full_features, prompts, calibration=calib,
                              style="numeric")
    assert "percentile" not in numeric  # numeric style = raw numbers only


def test_percentile_rank():
    from src.injection.render import percentile_rank

    vals = [0.1, 0.2, 0.3, 0.4]
    assert percentile_rank(vals, 0.05) == 0.0
    assert percentile_rank(vals, 0.25) == 50.0
    assert percentile_rank(vals, 0.9) == 100.0
    assert percentile_rank([], 0.5) == 50.0


def test_render_gates(full_features, prompts, cfg):
    import copy

    calib = {"sym_h": [0.1, 0.3, 0.5], "sym_v": [0.1, 0.3, 0.5],
             "abs_balance": [0.0, 0.2, 0.4],
             "gates": {"balance_min": 0.30, "sym_hi": 0.60, "sym_lo": 0.15,
                       "edge_only_when_near": True}}
    f = copy.deepcopy(full_features)
    f["balance"]["balance"] = -0.10          # weak signal -> line gated out
    f["symmetry"]["sym_h"] = 0.40            # mid zone -> line gated out
    f["subject"]["edge"].update(near_edge=False, spans_width=False,
                                spans_height=False)
    text = render_evidence(f, prompts, calibration=calib, use_gates=True)
    assert "Visual mass" not in text and "Mirror similarity" not in text
    assert "Edges:" not in text
    assert "Subject:" in text                # position line always stays

    f["balance"]["balance"] = -0.45          # strong signal -> line speaks
    f["symmetry"]["sym_h"] = 0.70            # extreme high -> line speaks
    f["subject"]["edge"]["near_edge"] = True
    text2 = render_evidence(f, prompts, calibration=calib, use_gates=True)
    assert "Visual mass" in text2 and "Mirror similarity" in text2
    assert "Edges:" in text2

    ungated = render_evidence(f, prompts, calibration=calib)  # default: no gates
    assert "Visual mass" in ungated


def test_render_numeric_style_strips_qualifiers(full_features, prompts):
    text = render_evidence(full_features, prompts, style="numeric")
    assert "thirds-aligned" not in text
    assert "concentrated" not in text and "similarity is high" not in text
    assert "SSIM" in text and "Visual mass:" in text  # numbers stay
    with pytest.raises(ValueError):
        render_evidence(full_features, prompts, style="fancy")


def test_lexicon_guard_raises(prompts):
    with pytest.raises(LexiconViolation):
        lexicon_guard("this is a rule of thirds composition",
                      prompts["banned_lexicon"])


def test_render_no_subject_branch(full_features, prompts):
    f = dict(full_features, no_subject=True, subject=None)
    text = render_evidence(f, prompts)
    assert "no single dominant subject" in text
    assert "Mirror similarity:" in text  # frame-level lines still present


def test_render_feature_groups(full_features, prompts):
    only_pos = render_evidence(full_features, prompts, include=GROUPS["position"])
    assert "Subject:" in only_pos and "Mirror similarity:" not in only_pos
    loo_sym = render_evidence(full_features, prompts,
                              include={"F1", "F2", "F3", "F6", "F7"})
    assert "Mirror similarity:" not in loo_sym and "Depth:" in loo_sym


def test_build_prompt_conditions(full_features, prompts):
    opts = {"A": "Alpha", "B": "Beta"}
    ev = render_evidence(full_features, prompts)
    c0 = build_prompt("Q?", opts, "C0", prompts)
    assert "Question: Q?" in c0 and "Options:\nA. Alpha" in c0
    assert "select the most likely answer" in c0
    assert "<composition_evidence>" not in c0
    c2 = build_prompt("Q?", opts, "C2", prompts, evidence_text=ev)
    assert c2.index("<composition_evidence>") < c2.index("Q?")
    c2_after = build_prompt("Q?", opts, "C2", prompts, evidence_text=ev,
                            position="after_question")
    assert c2_after.index("<composition_evidence>") > c2_after.index("Q?")
    with pytest.raises(ValueError):
        build_prompt("Q?", opts, "C2", prompts)  # evidence required


def test_perturb_position_consistent(full_features, cfg):
    import math

    p = perturb_features(full_features, "position", cfg["thresholds"], seed=1)
    s = p["subject"]
    assert (s["x"], s["y"]) != (full_features["subject"]["x"],
                                full_features["subject"]["y"])
    # internal consistency: d_center matches the new coordinates
    assert math.isclose(s["d_center"], math.hypot(s["x"] - 0.5, s["y"] - 0.5),
                        abs_tol=1e-9)


def test_perturb_balance_flip(full_features, cfg):
    p = perturb_features(full_features, "balance", cfg["thresholds"])
    assert p["balance"]["balance"] == -full_features["balance"]["balance"]
    assert p["balance"]["mass_left"] == full_features["balance"]["mass_right"]


def test_perturb_keeps_schema_and_renders(full_features, cfg, prompts):
    p = perturb_features(full_features, "all", cfg["thresholds"])
    assert set(p.keys()) == set(full_features.keys())
    assert render_evidence(p, prompts).startswith("<composition_evidence>")
