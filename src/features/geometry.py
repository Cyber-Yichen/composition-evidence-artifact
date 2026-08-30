from __future__ import annotations

import math

THIRDS_POINTS = [(1 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3), (2 / 3, 2 / 3)]
THIRDS_VLINES = [1 / 3, 2 / 3]


def box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def center_offset(x: float, y: float) -> dict:
    """F1: subject centroid and distance to frame center."""
    return {"x": x, "y": y, "d_center": math.hypot(x - 0.5, y - 0.5)}


def thirds_reference(x: float, y: float, thr_center: float, thr_thirds: float,
                     d_center: float | None = None) -> dict:
    """F2: distances to thirds intersections / vertical thirds lines + alignment judgment."""
    if d_center is None:
        d_center = math.hypot(x - 0.5, y - 0.5)
    dists = [math.hypot(x - px, y - py) for px, py in THIRDS_POINTS]
    i = min(range(4), key=lambda k: dists[k])
    d_vlines = min(abs(x - v) for v in THIRDS_VLINES)
    return {
        "d_thirds": dists[i],
        "nearest_thirds_point": list(THIRDS_POINTS[i]),
        "d_thirds_vlines": d_vlines,
        "center_aligned": d_center < thr_center,
        "thirds_aligned": dists[i] < thr_thirds,
    }


def area_ratio(box: list[float]) -> float:
    """F3: subject box area as a fraction of the frame."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def edge_distances(box: list[float], thr_near: float) -> dict:
    """F9: min normalized distance from the box to each frame edge."""
    x1, y1, x2, y2 = box
    d = {"left": x1, "right": 1.0 - x2, "top": y1, "bottom": 1.0 - y2}
    nearest = min(d, key=d.get)
    return {
        **d,
        "min_edge_dist": d[nearest],
        "nearest_edge": nearest,
        "near_edge": d[nearest] < thr_near,
        # a box touching BOTH opposite edges spans the frame; "near the top edge"
        # would be misleading wording for a full-height subject
        "spans_width": d["left"] < thr_near and d["right"] < thr_near,
        "spans_height": d["top"] < thr_near and d["bottom"] < thr_near,
    }
