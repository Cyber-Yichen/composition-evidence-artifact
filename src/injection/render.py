from __future__ import annotations

GROUPS = {
    "position": {"F1", "F2", "F3"},
    "sym_bal": {"F4", "F5"},
    "depth_lines": {"F6", "F7"},
    "extras": {"F8", "F9"},
}
ALL_FEATURES = {f for g in GROUPS.values() for f in g}


class LexiconViolation(ValueError):
    pass


def percentile_rank(sorted_vals: list[float], x: float) -> float:
    """Percentile of x within a sorted train-set value array (0-100)."""
    import bisect

    if not sorted_vals:
        return 50.0
    return 100.0 * bisect.bisect_left(sorted_vals, x) / len(sorted_vals)


def lexicon_guard(text: str, banned: list[str]) -> None:
    low = text.lower()
    hits = [b for b in banned if b.lower() in low]
    if hits:
        raise LexiconViolation(f"evidence text contains banned label terms: {hits}")


def _subject_line(f: dict, t: dict, include: set, neutral: bool) -> str | None:
    if f.get("no_subject") or f.get("subject") is None:
        return t["subject_none"]
    s = f["subject"]
    align_part = ""
    if "F2" in include:
        tmpl = (t["align_thirds"] if neutral and s["thirds_aligned"]
                else t["align_none"])
        align_part = tmpl.format(**s)
    pct = s["area_ratio"] * 100
    area_pct = f"{pct:.1f}" if pct < 10 else f"{pct:.0f}"  # keep tiny subjects visible
    return t["subject"].format(phrase=s["phrase"], x=s["x"], y=s["y"],
                               area_pct=area_pct, align_part=align_part)


def render_evidence(features: dict, prompts_cfg: dict,
                    include: set[str] | None = None, guard: bool = True,
                    style: str = "neutral",
                    calibration: dict | None = None,
                    use_gates: bool = False) -> str:
    """calibration: configs/calibration.json content (train-set sorted value arrays)
    — enables '(NNth percentile of film frames)' context on F4/F5 numbers.
    use_gates: apply train-fitted speak-only-when-confident gates
    (calibration["gates"], see scripts/fit_gates.py)."""
    if style not in ("neutral", "numeric"):
        raise ValueError(f"unknown evidence style {style}")
    neutral = style == "neutral"
    t = prompts_cfg["evidence"]
    include = ALL_FEATURES if include is None else set(include)
    lines: list[str] = []

    if include & GROUPS["position"]:
        line = _subject_line(features, t, include, neutral)
        if line:
            lines.append(line)

    gates = (calibration or {}).get("gates") if use_gates else None

    if "F9" in include and features.get("subject"):
        e = features["subject"]["edge"]
        flagged = e["near_edge"] or e.get("spans_height") or e.get("spans_width")
        if not (gates and gates.get("edge_only_when_near") and not flagged):
            flag = ""
            if neutral:
                if e.get("spans_height"):
                    flag = t["edge_flag_span_h"]
                elif e.get("spans_width"):
                    flag = t["edge_flag_span_w"]
                elif e["near_edge"]:
                    flag = t["edge_flag_near"].format(which=e["nearest_edge"])
            lines.append(t["edge"].format(left=e["left"], right=e["right"],
                                          top=e["top"], bottom=e["bottom"],
                                          edge_flag=flag))

    def pct(key: str, x: float) -> str:
        if not neutral or not calibration or key not in calibration:
            return ""
        return t["pct_part"].format(pct=percentile_rank(calibration[key], x))

    if "F5" in include and features.get("balance"):
        b = features["balance"]
        # gate: the balance line speaks only when the train-fitted confidence
        # threshold is met (weak mass signals drag centered/even compositions)
        if not (gates and abs(b["balance"]) < gates["balance_min"]):
            skew = t["skew_none"]
            if neutral and b["skew"] == "left":
                skew = t["skew_left"]
            elif neutral and b["skew"] == "right":
                skew = t["skew_right"]
            lines.append(t["balance"].format(
                mass_left=b["mass_left"], mass_right=b["mass_right"],
                absdiff=abs(b["balance"]),
                bal_pct=pct("abs_balance", abs(b["balance"])), skew_part=skew))

    if "F4" in include and features.get("symmetry"):
        s = features["symmetry"]
        # gate: mirror line only at informative extremes (high or clearly low)
        if not (gates and gates["sym_lo"] < s["sym_h"] < gates["sym_hi"]):
            lines.append(t["symmetry"].format(sym_h=s["sym_h"], sym_v=s["sym_v"],
                                              sym_h_pct=pct("sym_h", s["sym_h"]),
                                              sym_v_pct=pct("sym_v", s["sym_v"])))

    if "F6" in include and features.get("depth"):
        d = features["depth"]
        r = d["ratios"]
        vals = dict(fg_pct=r["foreground"] * 100, mid_pct=r["mid"] * 100,
                    bg_pct=r["background"] * 100)
        if d["subject_layer"]:
            lines.append(t["depth"].format(subject_layer=d["subject_layer"], **vals))
        else:
            lines.append(t["depth_no_subject"].format(**vals))

    if "F7" in include and features.get("lines"):
        ln = features["lines"]
        angle_part = (t["angle_part"].format(angle=ln["dominant_angle_deg"])
                      if ln.get("dominant_angle_deg") is not None else "")
        horizon_part = (t["horizon_part"].format(horizon_y=ln["horizon_y"])
                        if ln.get("horizon_y") is not None else "")
        tmpl = t["lines"] if features.get("subject") else t["lines_no_subject"]
        lines.append(tmpl.format(n_leading=ln["n_leading"], n_top=ln["n_top"],
                                 angle_part=angle_part, horizon_part=horizon_part))

    if "F8" in include and features.get("headroom", {}).get("face_top_y") is not None:
        h = features["headroom"]
        lines.append(t["headroom"].format(face_top=h["face_top_y"],
                                          headroom_pct=h["headroom"] * 100))

    block = "\n".join([t["header"], *lines, t["footer"], t["note"]])
    if guard:
        lexicon_guard(block, prompts_cfg["banned_lexicon"])
    return block
