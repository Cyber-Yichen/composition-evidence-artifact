from __future__ import annotations

import math


def _dilate_box(box: list[float], factor: float) -> list[float]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2 * factor, (y2 - y1) / 2 * factor
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def _line_intersects_box(p1: list[float], p2: list[float], box: list[float]) -> bool:
    """Infinite line through p1,p2 vs axis-aligned box: corners must not all lie
    strictly on the same side of the line."""
    x1, y1, x2, y2 = box
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return False
    sides = [dx * (cy - p1[1]) - dy * (cx - p1[0])
             for cx, cy in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]]
    return min(sides) <= 1e-12 and max(sides) >= -1e-12


def _axial_mean_deg(angles_deg: list[float], weights: list[float]) -> float:
    """Length-weighted mean of axial (mod-180) angles via angle doubling."""
    s = sum(w * math.sin(math.radians(2 * a)) for a, w in zip(angles_deg, weights))
    c = sum(w * math.cos(math.radians(2 * a)) for a, w in zip(angles_deg, weights))
    a = math.degrees(math.atan2(s, c)) / 2
    return a + 180 if a < -90 else (a - 180 if a >= 90 else a)


def leading_lines(segments: list[dict], subject_box: list[float] | None,
                  top_k: int = 10, dilation: float = 1.2) -> dict:
    top = sorted(segments, key=lambda s: s["length"], reverse=True)[:top_k]
    out = {"n_top": len(top), "n_leading": 0, "dominant_angle_deg": None}
    if not top:
        return out
    if subject_box is None:
        # no-subject branch: report dominant orientation of the long lines instead
        out["dominant_angle_deg"] = _axial_mean_deg(
            [s["angle_deg"] for s in top], [s["length"] for s in top])
        return out
    dbox = _dilate_box(subject_box, dilation)
    leading = [s for s in top if _line_intersects_box(s["p1"], s["p2"], dbox)]
    out["n_leading"] = len(leading)
    if leading:
        out["dominant_angle_deg"] = _axial_mean_deg(
            [s["angle_deg"] for s in leading], [s["length"] for s in leading])
    return out


def horizon(segments: list[dict], angle_tol_deg: float = 10.0,
            min_len: float = 0.15) -> dict:
    """Longest sufficiently-long near-horizontal segment -> its mean y."""
    cands = [s for s in segments
             if min(abs(s["angle_deg"]), 180 - abs(s["angle_deg"])) <= angle_tol_deg
             and s["length"] >= min_len]
    if not cands:
        return {"horizon_y": None}
    best = max(cands, key=lambda s: s["length"])
    return {"horizon_y": (best["p1"][1] + best["p2"][1]) / 2}
