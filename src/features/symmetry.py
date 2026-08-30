from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity
from skimage.transform import resize


def _prep(gray: np.ndarray, resize_to: int) -> np.ndarray:
    g = gray.astype(np.float64)
    if g.max() > 1.0:
        g = g / 255.0
    h, w = g.shape
    if max(h, w) > resize_to:
        s = resize_to / max(h, w)
        g = resize(g, (max(8, int(h * s)), max(8, int(w * s))), anti_aliasing=True)
    return g


def mirror_symmetry(gray: np.ndarray, thr_strong: float, resize_to: int = 256) -> dict:
    g = _prep(gray, resize_to)
    sym_h = float(structural_similarity(g, np.fliplr(g), data_range=1.0))
    sym_v = float(structural_similarity(g, np.flipud(g), data_range=1.0))
    return {
        "sym_h": sym_h,
        "sym_v": sym_v,
        "strong_h": sym_h > thr_strong,
        "strong_v": sym_v > thr_strong,
    }
