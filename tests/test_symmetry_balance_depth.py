import numpy as np

from src.features.balance import lr_balance
from src.features.depth_layers import depth_layers
from src.features.symmetry import mirror_symmetry


# ---- F4 ----

def test_f4_perfect_lr_mirror():
    rng = np.random.default_rng(42)
    half = rng.integers(0, 255, (120, 80), dtype=np.uint8)
    img = np.hstack([half, np.fliplr(half)])
    f = mirror_symmetry(img, thr_strong=0.7)
    assert f["sym_h"] > 0.95 and f["strong_h"]


def test_f4_asymmetric_low():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (120, 160), dtype=np.uint8)
    f = mirror_symmetry(img, thr_strong=0.7)
    assert f["sym_h"] < 0.7 and not f["strong_h"]


def test_f4_vertical_mirror():
    rng = np.random.default_rng(7)
    half = rng.integers(0, 255, (60, 160), dtype=np.uint8)
    img = np.vstack([half, np.flipud(half)])
    f = mirror_symmetry(img, thr_strong=0.7)
    assert f["sym_v"] > 0.95 and f["strong_v"]


# ---- F5 ----

def test_f5_left_heavy_mass():
    sal = np.zeros((100, 200), dtype=np.float32)
    sal[:, :60] = 1.0
    f = lr_balance(sal, thr_skew=0.25)
    assert f["balance"] < -0.9 and f["skew"] == "left"
    assert f["mass_left"] > 0.95


def test_f5_right_heavy_mass():
    sal = np.zeros((100, 200), dtype=np.float32)
    sal[:, 150:] = 1.0
    f = lr_balance(sal, thr_skew=0.25)
    assert f["balance"] > 0.9 and f["skew"] == "right"


def test_f5_even_mass_no_skew():
    sal = np.ones((100, 201), dtype=np.float32)  # odd width: middle col excluded
    f = lr_balance(sal, thr_skew=0.25)
    assert abs(f["balance"]) < 1e-6 and f["skew"] is None


# ---- F6 ----

def _piecewise_closeness() -> np.ndarray:
    # 60% far (0.1), 30% mid (0.5), 10% near (0.9) — ratios must reflect this
    close = np.empty((100, 100), dtype=np.float32)
    close[:, :60] = 0.1
    close[:, 60:90] = 0.5
    close[:, 90:] = 0.9
    return close


def test_f6_equal_width_bins_reflect_scene():
    f = depth_layers(_piecewise_closeness(), subject_box=[0.92, 0.4, 0.98, 0.6])
    assert f["subject_layer"] == "foreground"
    assert abs(f["ratios"]["background"] - 0.6) < 0.01
    assert abs(f["ratios"]["mid"] - 0.3) < 0.01
    assert abs(f["ratios"]["foreground"] - 0.1) < 0.01


def test_f6_subject_background():
    f = depth_layers(_piecewise_closeness(), subject_box=[0.05, 0.4, 0.3, 0.6])
    assert f["subject_layer"] == "background"


def test_f6_flat_depth_degenerate():
    close = np.full((100, 100), 0.5, dtype=np.float32)
    f = depth_layers(close, subject_box=[0.4, 0.4, 0.6, 0.6])
    assert f["degenerate"] and f["subject_layer"] == "mid-ground"
