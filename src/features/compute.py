from __future__ import annotations

import numpy as np

from src.extractors.saliency import make_pseudo_saliency
from src.features import geometry, subject_selection
from src.features.balance import lr_balance
from src.features.depth_layers import depth_layers
from src.features.headroom import headroom
from src.features.lines import horizon, leading_lines
from src.features.symmetry import mirror_symmetry
from src.utils import to_jsonable


def compute_features(extraction: dict, cfg: dict) -> dict:
    thr = cfg["thresholds"]
    boxes = extraction.get("boxes") or []
    gray = extraction["gray"]
    closeness = extraction.get("closeness")
    segments = extraction.get("segments") or []

    sel = subject_selection.select_dominant(boxes, cfg["subject_selection"])
    subject = sel["subject"]
    out: dict = {"no_subject": sel["no_subject"], "subject": None}

    if subject is not None:
        box = subject["box"]
        x, y = geometry.box_center(box)
        f1 = geometry.center_offset(x, y)
        f2 = geometry.thirds_reference(x, y, thr["center_align"], thr["thirds_align"],
                                       d_center=f1["d_center"])
        out["subject"] = {
            "phrase": subject["phrase"],
            "box": box,
            **f1,
            **f2,
            "area_ratio": geometry.area_ratio(box),                 # F3
            "edge": geometry.edge_distances(box, thr["edge_near"]),  # F9
        }

    # F4 symmetry (frame-level, always available)
    out["symmetry"] = mirror_symmetry(gray, thr["symmetry_strong"],
                                      cfg["features"]["symmetry_resize"])

    # F5 balance on pseudo-saliency
    sal_cfg = cfg["extractors"]["saliency"]
    sal = make_pseudo_saliency(gray.shape[0], gray.shape[1], boxes, closeness,
                               sal_cfg["w_boxes"], sal_cfg["w_closeness"])
    out["balance"] = lr_balance(sal, thr["balance_skew"])

    # F6 depth layers
    out["depth"] = (depth_layers(closeness, subject["box"] if subject else None)
                    if closeness is not None else None)

    # F7 lines
    lcfg = cfg["extractors"]["lines"]
    out["lines"] = {
        **leading_lines(segments, subject["box"] if subject else None,
                        lcfg["top_k"], cfg["features"]["leading_line_box_dilation"]),
        **horizon(segments, lcfg["horizon_angle_tol_deg"], lcfg["min_horizon_len"]),
    }

    # F8 headroom
    out["headroom"] = headroom(sel["face"]["box"] if sel["face"] else None)

    out["meta"] = {
        "n_boxes": len(boxes),
        "height": int(gray.shape[0]),
        "width": int(gray.shape[1]),
        "has_depth": closeness is not None,
        "n_segments": len(segments),
    }
    return to_jsonable(out)
