from __future__ import annotations

import math

MAX_CENTER_DIST = math.hypot(0.5, 0.5)  # corner-to-center distance in normalized coords


def _is_person(phrase: str, person_terms: list[str]) -> bool:
    p = phrase.lower()
    return any(t in p for t in person_terms)


def subject_score(box: dict, cfg: dict) -> float:
    x1, y1, x2, y2 = box["box"]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    center_prior = 1.0 - math.hypot(cx - 0.5, cy - 0.5) / MAX_CENTER_DIST
    score = (
        cfg["w_area"] * area
        + cfg["w_center_prior"] * center_prior
        + cfg["w_confidence"] * box["score"]
    )
    if _is_person(box["phrase"], cfg["person_terms"]):
        score *= cfg["person_boost"]
    return score


def _box_area(box: dict) -> float:
    x1, y1, x2, y2 = box["box"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def select_dominant(boxes: list[dict], cfg: dict) -> dict:
    """Returns {"subject": box|None, "face": box|None, "no_subject": bool}.

    face = highest-scoring box whose phrase contains "face" (for F8 headroom).
    Boxes covering more than max_area of the frame are excluded as subjects —
    a whole-frame detection carries no positional information.
    """
    max_area = cfg.get("max_area", 1.0)
    candidates = [b for b in boxes if _box_area(b) <= max_area]
    if not candidates:
        return {"subject": None, "face": None, "no_subject": True}
    scored = sorted(candidates, key=lambda b: subject_score(b, cfg), reverse=True)
    faces = [b for b in scored if "face" in b["phrase"].lower()]
    return {
        "subject": scored[0],
        "face": faces[0] if faces else None,
        "no_subject": False,
    }
