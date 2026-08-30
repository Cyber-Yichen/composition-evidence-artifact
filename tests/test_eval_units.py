import math

import pytest

from src.eval.metrics import compare_conditions, flip_matrix, mcnemar_exact
from src.eval.parsing import parse_choice
from src.extractors.frames import frame_indices
from src.injection.soft_token import FEATURE_DIM, featurize

OPTS = {"A": "Extreme Close Up", "B": "Medium Close Up",
        "C": "Close Up", "D": "Medium Wide"}


@pytest.mark.parametrize("text,expected", [
    ("D", "D"),
    ("D.", "D"),
    ("(B)", "B"),
    ("**A**", "A"),
    ("D. Medium Wide", "D"),
    ("The answer is C.", "C"),
    ("Answer: B", "B"),
    ("I think the composition suggests...\nFinal answer: A", "A"),
    ("Let me think step by step.\nC", "C"),
    ("The shot is a Medium Wide.", "D"),          # unique option-text match
    ("", None),
    ("<think>long reasoning here...</think>\n<answer>B</answer>", "B"),
    ("<think>x</think><answer> C. Close Up </answer>", "C"),
])
def test_parse_choice(text, expected):
    letter, _ = parse_choice(text, OPTS)
    assert letter == expected


def test_parse_ambiguous_fails():
    letter, method = parse_choice(
        "It could be Close Up or Medium Close Up honestly", OPTS)
    assert letter is None and method == "fail"


def test_mcnemar_known_value():
    # b=15, c=5 discordant -> two-sided exact p ≈ 0.0414
    assert math.isclose(mcnemar_exact(15, 5), 0.041389, abs_tol=1e-4)
    assert mcnemar_exact(0, 0) == 1.0


def test_flip_matrix_and_compare():
    base = {"u1": True, "u2": False, "u3": False, "u4": True}
    new = {"u1": True, "u2": True, "u3": False, "u4": False}
    fm = flip_matrix(base, new)
    assert fm == {"n": 4, "both_right": 1, "both_wrong": 1, "lost": 1, "gained": 1}
    recs_b = [{"uid": u, "correct": v} for u, v in base.items()]
    recs_n = [{"uid": u, "correct": v} for u, v in new.items()]
    c = compare_conditions(recs_b, recs_n)
    assert c["acc_base"] == 0.5 and c["acc_new"] == 0.5 and c["p_mcnemar"] == 1.0


def test_shotbench_loader_and_sc_subset():
    from src.eval.datasets import load_shotbench, sc_subset
    from src.utils import REPO_ROOT

    samples = load_shotbench(REPO_ROOT / "data" / "shotbench")
    assert len(samples) == 3572
    sc = sc_subset(samples)
    assert len(sc) == 479
    assert all(s.media_type == "image" for s in sc)
    s = sc[0]
    assert s.answer in s.options and s.media_paths[0].startswith("image/")


def test_featurize_dim_and_defaults(full_features):
    v = featurize(full_features)
    assert v.shape == (FEATURE_DIM,)
    empty = featurize({"no_subject": True})
    assert empty.shape == (FEATURE_DIM,) and empty[22] == 0.0  # has_subject flag


def test_frame_indices_uniform_plus_middle():
    idxs, middle = frame_indices(101, n_uniform=4)
    assert middle == 50 and 50 in idxs and 0 in idxs and 100 in idxs
    assert len(idxs) == 5
    assert frame_indices(0)[0] == []
