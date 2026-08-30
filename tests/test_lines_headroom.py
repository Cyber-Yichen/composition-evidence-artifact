import math

from src.features.headroom import headroom
from src.features.lines import horizon, leading_lines
from tests.conftest import seg


def test_f7_converging_lines_counted():
    box = [0.3, 0.3, 0.5, 0.5]
    segs = [
        seg(0.0, 1.0, 0.35, 0.45),   # aims into the box
        seg(1.0, 1.0, 0.45, 0.45),   # aims into the box
        seg(0.0, 0.05, 1.0, 0.05),   # far horizontal, extension misses the box
    ]
    f = leading_lines(segs, box, top_k=10, dilation=1.2)
    assert f["n_top"] == 3
    assert f["n_leading"] == 2


def test_f7_dominant_angle_weighted_mean():
    box = [0.4, 0.4, 0.6, 0.6]
    s1 = seg(0.0, 0.5 - math.tan(math.radians(30)) * 0.5, 1.0,
             0.5 + math.tan(math.radians(30)) * 0.5)  # ~30 deg through center
    s2 = seg(0.0, 0.5 - math.tan(math.radians(40)) * 0.5, 1.0,
             0.5 + math.tan(math.radians(40)) * 0.5)  # ~40 deg through center
    f = leading_lines([s1, s2], box)
    assert 30 <= f["dominant_angle_deg"] <= 40


def test_f7_no_subject_reports_orientation_only():
    f = leading_lines([seg(0.0, 0.0, 1.0, 0.0)], None)
    assert f["n_leading"] == 0
    assert abs(f["dominant_angle_deg"]) < 1e-6


def test_f7_no_segments():
    f = leading_lines([], [0.4, 0.4, 0.6, 0.6])
    assert f["n_top"] == 0 and f["dominant_angle_deg"] is None


def test_f7_horizon_longest_horizontal():
    segs = [seg(0.1, 0.62, 0.9, 0.62),        # long horizontal at y=0.62
            seg(0.4, 0.2, 0.5, 0.21),         # short: below min_len
            seg(0.0, 0.0, 1.0, 1.0)]          # diagonal: outside angle tol
    f = horizon(segs, angle_tol_deg=10, min_len=0.15)
    assert math.isclose(f["horizon_y"], 0.62, abs_tol=1e-9)


def test_f7_horizon_none():
    assert horizon([seg(0.0, 0.0, 1.0, 1.0)])["horizon_y"] is None


def test_f8_headroom():
    f = headroom([0.45, 0.12, 0.55, 0.3])
    assert math.isclose(f["headroom"], 0.12)


def test_f8_no_face():
    f = headroom(None)
    assert f["headroom"] is None and f["face_top_y"] is None
