from __future__ import annotations

import numpy as np


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def make_pseudo_saliency(h: int, w: int, boxes: list[dict],
                         closeness: np.ndarray | None,
                         w_boxes: float = 0.5, w_closeness: float = 0.5) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.float32)
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        c0, c1 = max(0, int(x1 * w)), min(w, max(int(x2 * w), int(x1 * w) + 1))
        r0, r1 = max(0, int(y1 * h)), min(h, max(int(y2 * h), int(y1 * h) + 1))
        region = mask[r0:r1, c0:c1]
        np.maximum(region, float(b["score"]), out=region)

    close = np.zeros((h, w), dtype=np.float32)
    if closeness is not None:
        c = np.asarray(closeness, dtype=np.float32)
        if c.shape != (h, w):
            import cv2
            c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
        close = _norm(c)

    if not boxes and closeness is None:
        return np.full((h, w), 1.0 / (h * w), dtype=np.float32)  # uniform, balance -> 0

    sal = w_boxes * _norm(mask) + w_closeness * close
    return sal.astype(np.float32)
