import math

from src.features import geometry


def test_f1_centered():
    f = geometry.center_offset(0.5, 0.5)
    assert f["d_center"] == 0.0


def test_f1_offset_distance():
    f = geometry.center_offset(0.8, 0.5)
    assert math.isclose(f["d_center"], 0.3, abs_tol=1e-9)


def test_f2_center_aligned():
    f = geometry.thirds_reference(0.52, 0.49, thr_center=0.08, thr_thirds=0.08)
    assert f["center_aligned"] and not f["thirds_aligned"]


def test_f2_thirds_aligned():
    f = geometry.thirds_reference(1 / 3, 1 / 3, thr_center=0.08, thr_thirds=0.08)
    assert f["thirds_aligned"] and not f["center_aligned"]
    assert f["d_thirds"] == 0.0
    assert f["nearest_thirds_point"] == [1 / 3, 1 / 3]


def test_f2_neither_aligned():
    f = geometry.thirds_reference(0.05, 0.05, thr_center=0.08, thr_thirds=0.08)
    assert not f["center_aligned"] and not f["thirds_aligned"]


def test_f2_vline_distance():
    f = geometry.thirds_reference(0.4, 0.5, 0.08, 0.08)
    assert math.isclose(f["d_thirds_vlines"], abs(0.4 - 1 / 3), abs_tol=1e-9)


def test_f3_area():
    assert math.isclose(geometry.area_ratio([0.25, 0.25, 0.75, 0.75]), 0.25)
    assert geometry.area_ratio([0.5, 0.5, 0.5, 0.9]) == 0.0  # degenerate box


def test_f9_edge_near_left():
    f = geometry.edge_distances([0.01, 0.3, 0.2, 0.7], thr_near=0.05)
    assert f["near_edge"] and f["nearest_edge"] == "left"
    assert math.isclose(f["min_edge_dist"], 0.01)


def test_f9_centered_not_near():
    f = geometry.edge_distances([0.4, 0.35, 0.6, 0.65], thr_near=0.05)
    assert not f["near_edge"]
    assert math.isclose(f["left"], 0.4) and math.isclose(f["right"], 0.4)
    assert not f["spans_width"] and not f["spans_height"]


def test_f9_full_height_box_spans():
    f = geometry.edge_distances([0.3, 0.0, 0.55, 1.0], thr_near=0.05)
    assert f["spans_height"] and not f["spans_width"]
