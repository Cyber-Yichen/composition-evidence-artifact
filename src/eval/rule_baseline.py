from __future__ import annotations

BASE_ORDER = ["Symmetrical", "Left heavy", "Right heavy",
              "Short side", "Center", "Balanced"]


def ranked_labels(features: dict, thr: dict) -> list[str]:
    s = features.get("subject") or {}
    sym = features.get("symmetry") or {}
    bal = (features.get("balance") or {}).get("balance", 0.0)

    ranked: list[str] = []
    if sym.get("sym_h", 0.0) > thr["symmetry_strong"]:
        ranked.append("Symmetrical")
    if abs(bal) > thr["balance_skew"]:
        ranked.append("Left heavy" if bal < 0 else "Right heavy")
    if s and s.get("edge", {}).get("min_edge_dist", 1.0) < thr["edge_near"]:
        ranked.append("Short side")
    if s and s.get("d_center", 1.0) < thr["center_align"]:
        ranked.append("Center")
    ranked.append("Balanced")
    ranked += [l for l in BASE_ORDER if l not in ranked]
    return ranked


def rule_choice(features: dict, options: dict[str, str], thr: dict) -> str:
    """Pick an option letter deterministically from the ranked labels."""
    for label in ranked_labels(features, thr):
        hits = sorted(k for k, v in options.items() if label in v)
        if hits:
            # prefer the exact single-label option over a combo containing it
            exact = [k for k in hits if options[k].strip() == label]
            return exact[0] if exact else hits[0]
    return sorted(options)[0]  # unreachable with ShotBench SC vocab; deterministic
