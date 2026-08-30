from __future__ import annotations

import numpy as np

LO, HI = 1 / 3, 2 / 3  # equal-width bin edges on [0, 1]


def _layer_of(value: float) -> str:
    if value < LO:
        return "background"
    if value >= HI:
        return "foreground"
    return "mid-ground"


def depth_layers(closeness: np.ndarray, subject_box: list[float] | None = None) -> dict:
    c = np.asarray(closeness, dtype=np.float32)
    degenerate = bool(float(c.std()) < 1e-6)
    if degenerate:
        ratios = {"foreground": 0.0, "mid": 1.0, "background": 0.0}
        subject_layer = "mid-ground" if subject_box is not None else None
        return {"subject_layer": subject_layer, "ratios": ratios, "degenerate": True}

    bg = float((c < LO).mean())
    fg = float((c >= HI).mean())
    ratios = {"foreground": fg, "mid": 1.0 - fg - bg, "background": bg}

    subject_layer = None
    if subject_box is not None:
        h, w = c.shape
        x1, y1, x2, y2 = subject_box
        c0, c1 = max(0, int(x1 * w)), min(w, max(int(x2 * w), int(x1 * w) + 1))
        r0, r1 = max(0, int(y1 * h)), min(h, max(int(y2 * h), int(y1 * h) + 1))
        subject_layer = _layer_of(float(np.median(c[r0:r1, c0:c1])))

    return {"subject_layer": subject_layer, "ratios": ratios, "degenerate": False}
