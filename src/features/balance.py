from __future__ import annotations

import numpy as np

EPS = 1e-9


def lr_balance(saliency: np.ndarray, thr_skew: float) -> dict:
    h, w = saliency.shape
    half = w // 2
    left = float(saliency[:, :half].sum())
    right = float(saliency[:, w - half:].sum())  # symmetric halves; middle col of odd widths excluded
    total = left + right + EPS
    balance = (right - left) / total
    skew = None
    if abs(balance) > thr_skew:
        skew = "left" if balance < 0 else "right"
    return {
        "balance": balance,
        "mass_left": left / total,
        "mass_right": right / total,
        "skew": skew,
    }
