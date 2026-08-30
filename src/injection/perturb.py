from __future__ import annotations

import copy
import random

from src.features import geometry

MODES = ["position", "balance", "symmetry", "all"]


def _perturb_position(f: dict, rng: random.Random, thr: dict) -> None:
    s = f.get("subject")
    if not s:
        return
    w = s["box"][2] - s["box"][0]
    h = s["box"][3] - s["box"][1]
    x = rng.uniform(0.05, 0.95)
    y = rng.uniform(0.05, 0.95)
    s["box"] = [max(0.0, x - w / 2), max(0.0, y - h / 2),
                min(1.0, x + w / 2), min(1.0, y + h / 2)]
    s.update(geometry.center_offset(x, y))
    s.update(geometry.thirds_reference(x, y, thr["center_align"],
                                       thr["thirds_align"], d_center=s["d_center"]))
    s["edge"] = geometry.edge_distances(s["box"], thr["edge_near"])


def _perturb_balance(f: dict, thr: dict) -> None:
    b = f.get("balance")
    if not b:
        return
    b["balance"] = -b["balance"]
    b["mass_left"], b["mass_right"] = b["mass_right"], b["mass_left"]
    b["skew"] = (None if abs(b["balance"]) <= thr["balance_skew"]
                 else ("left" if b["balance"] < 0 else "right"))


def _perturb_symmetry(f: dict, rng: random.Random, thr: dict) -> None:
    s = f.get("symmetry")
    if not s:
        return
    s["sym_h"] = rng.uniform(0.0, 1.0)
    s["sym_v"] = rng.uniform(0.0, 1.0)
    s["strong_h"] = s["sym_h"] > thr["symmetry_strong"]
    s["strong_v"] = s["sym_v"] > thr["symmetry_strong"]


def perturb_features(features: dict, mode: str, thresholds: dict,
                     seed: int = 42) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown perturbation mode {mode}")
    rng = random.Random(seed)
    f = copy.deepcopy(features)
    if mode in ("position", "all"):
        _perturb_position(f, rng, thresholds)
    if mode in ("balance", "all"):
        _perturb_balance(f, thresholds)
    if mode in ("symmetry", "all"):
        _perturb_symmetry(f, rng, thresholds)
    return f
