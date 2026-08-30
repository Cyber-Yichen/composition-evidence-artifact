from __future__ import annotations

import numpy as np
import pytest

from src.utils import load_config, load_prompts


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def prompts():
    return load_prompts()


def make_box(cx: float, cy: float, w: float = 0.2, h: float = 0.3,
             score: float = 0.9, phrase: str = "person") -> dict:
    return {"box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            "score": score, "phrase": phrase}


def gray_with_rect(H: int = 200, W: int = 300, box: dict | None = None,
                   bg: int = 20, fg: int = 220) -> np.ndarray:
    img = np.full((H, W), bg, dtype=np.uint8)
    if box is not None:
        x1, y1, x2, y2 = box["box"]
        img[int(y1 * H):int(y2 * H), int(x1 * W):int(x2 * W)] = fg
    return img


def seg(x1, y1, x2, y2) -> dict:
    import math

    length = math.hypot(x2 - x1, y2 - y1)
    a = math.degrees(math.atan2(y2 - y1, x2 - x1))
    a = a - 180 if a >= 90 else (a + 180 if a < -90 else a)
    return {"p1": [x1, y1], "p2": [x2, y2], "length": length, "angle_deg": a}


@pytest.fixture()
def full_features(cfg):
    """A realistic complete feature dict for render/perturb/soft-token tests."""
    from src.features.compute import compute_features

    box = make_box(1 / 3, 1 / 3)
    gray = gray_with_rect(box=box)
    H, W = gray.shape
    yy = np.linspace(0, 1, H)[:, None]
    closeness = np.repeat(yy, W, axis=1)  # bottom of frame closer, like a ground plane
    segments = [seg(0.0, 0.62, 1.0, 0.62), seg(0.0, 1.0, 1 / 3, 1 / 3),
                seg(1.0, 1.0, 1 / 3, 1 / 3)]
    return compute_features({"boxes": [box, make_box(0.8, 0.8, 0.1, 0.1, 0.4, "car")],
                             "gray": gray, "closeness": closeness,
                             "segments": segments}, cfg)
