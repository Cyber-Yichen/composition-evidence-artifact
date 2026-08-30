import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_thresholds import youden_threshold  # noqa: E402


def test_youden_perfect_separation():
    fit = youden_threshold(np.array([0.8, 0.9, 0.85]), np.array([0.1, 0.2, 0.3]))
    assert fit["j"] == 1.0
    assert 0.3 < fit["threshold"] <= 0.8
    assert fit["tpr"] == 1.0 and fit["fpr"] == 0.0


def test_youden_overlapping():
    pos = np.array([0.5, 0.6, 0.7, 0.8])
    neg = np.array([0.3, 0.4, 0.5, 0.6])
    fit = youden_threshold(pos, neg)
    assert 0 < fit["j"] < 1
    assert fit["threshold"] >= 0.5  # cuts inside the overlap, above neg majority


def test_youden_no_signal():
    same = np.array([0.4, 0.5, 0.6])
    fit = youden_threshold(same, same)
    assert abs(fit["j"]) < 1e-9
