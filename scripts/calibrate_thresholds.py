from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

BASE_LABELS = ["Center", "Symmetrical", "Balanced", "Left heavy", "Right heavy",
               "Short side"]
SC_QUESTION_MARK = "composition"

# sft.json contains items of
# {"images": ["data/XXXX.jpg"], "messages": [{role, content}...]} — multi-turn,
# one dimension question per user turn, assistant replies with the letter.
COMP_QUESTION = "What's the composition of this shot?"


def youden_threshold(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Best 1-D threshold separating pos (should be ABOVE thr) from neg,
    maximizing Youden J = TPR - FPR. Pure function, unit-tested."""
    values = np.unique(np.concatenate([pos, neg]))
    best = {"threshold": float(values[0]), "j": -1.0, "tpr": 0.0, "fpr": 0.0}
    for t in values:
        tpr = float((pos >= t).mean())
        fpr = float((neg >= t).mean())
        j = tpr - fpr
        if j > best["j"]:
            best = {"threshold": float(t), "j": j, "tpr": tpr, "fpr": fpr}
    return best


def load_shotqa_sc(root: Path) -> list[dict]:
    """Returns [{image, label}] for single-label composition items."""
    import re

    data = json.load(open(root / "sft.json", encoding="utf-8"))
    out = []
    for item in data:
        msgs = item.get("messages", [])
        imgs = item.get("images", [])
        if not imgs:
            continue
        for i, m in enumerate(msgs):
            if m.get("role") != "user" or COMP_QUESTION not in m.get("content", ""):
                continue
            opts = dict(re.findall(r"([A-F])\.\s*([^\n]+)", m["content"]))
            if i + 1 < len(msgs) and msgs[i + 1].get("role") == "assistant":
                letter = msgs[i + 1]["content"].strip()[:1]
                ans = opts.get(letter, "").strip()
                if ans in BASE_LABELS:  # single-label only; combos excluded
                    out.append({"image": imgs[0], "label": ans})
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", action="store_true", help="print first items and exit")
    ap.add_argument("--per-label", type=int, default=300)
    args = ap.parse_args()

    from src.utils import load_config, set_seed

    cfg = load_config()
    set_seed(cfg["seed"])
    root = REPO / cfg["paths"]["shotqa"]

    if args.peek:
        data = json.load(open(root / "sft.json", encoding="utf-8"))
        print(f"items: {len(data)}")
        print(json.dumps(data[0], ensure_ascii=False, indent=2)[:1200])
        return

    import cv2

    from src.extractors.depth import DepthExtractor
    from src.extractors.grounding_dino import GroundingDINOExtractor
    from src.extractors.saliency import make_pseudo_saliency
    from src.features.balance import lr_balance
    from src.features.symmetry import mirror_symmetry
    from src.utils import file_key

    items = load_shotqa_sc(root)
    print(f"ShotQA single-label SC items: {len(items)}")
    by_label: dict[str, list] = {}
    rng = random.Random(cfg["seed"])
    for it in items:
        by_label.setdefault(it["label"], []).append(it)
    sample = []
    for label, lst in by_label.items():
        rng.shuffle(lst)
        sample += lst[: args.per_label]
        print(f"  {label}: {len(lst)} total, {min(len(lst), args.per_label)} sampled")

    g = cfg["extractors"]["grounding_dino"]
    d = cfg["extractors"]["depth"]
    cache = REPO / cfg["paths"]["cache"]
    dino = GroundingDINOExtractor(cache, g["model_id"], g["prompts"],
                                  g["box_threshold"], g["text_threshold"])
    depth = DepthExtractor(cache, d["model_id"], d["larger_is_closer"])
    sal_cfg = cfg["extractors"]["saliency"]

    rows = []
    for it in sample:
        # sft.json paths carry a "data/" prefix but the tar extracts flat
        p = root / it["image"]
        if not p.exists():
            p = root / Path(it["image"]).name
        img = cv2.imread(str(p))
        if img is None:
            continue
        key = file_key(p)
        boxes = dino(img, key)["boxes"]
        close = depth(img, key)["closeness"].astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sym = mirror_symmetry(gray, cfg["thresholds"]["symmetry_strong"],
                              cfg["features"]["symmetry_resize"])
        sal = make_pseudo_saliency(gray.shape[0], gray.shape[1], boxes, close,
                                   sal_cfg["w_boxes"], sal_cfg["w_closeness"])
        bal = lr_balance(sal, cfg["thresholds"]["balance_skew"])
        rows.append({"label": it["label"], "sym_h": sym["sym_h"],
                     "sym_v": sym["sym_v"], "balance": bal["balance"]})

    sym_pos = np.array([r["sym_h"] for r in rows if r["label"] == "Symmetrical"])
    sym_neg = np.array([r["sym_h"] for r in rows if r["label"] != "Symmetrical"])
    heavy = np.array([abs(r["balance"]) for r in rows
                      if r["label"] in ("Left heavy", "Right heavy")])
    balanced = np.array([abs(r["balance"]) for r in rows if r["label"] == "Balanced"])
    sign_ok = np.mean([
        (r["balance"] < 0) == (r["label"] == "Left heavy")
        for r in rows if r["label"] in ("Left heavy", "Right heavy")])

    sym_fit = youden_threshold(sym_pos, sym_neg)
    bal_fit = youden_threshold(heavy, balanced)

    rep = [
        "# F4/F5 threshold calibration report (ShotQA training split)", "",
        f"- Samples: {len(rows)} rows (seed={cfg['seed']}, at most {args.per_label})",
        f"- **Recommended symmetry_strong: {sym_fit['threshold']:.3f}** (previous 0.7; "
        f"J={sym_fit['j']:.3f}, TPR={sym_fit['tpr']:.3f}, FPR={sym_fit['fpr']:.3f})",
        f"- sym_h distribution: Symmetrical median={np.median(sym_pos):.3f}, "
        f"other labels median={np.median(sym_neg):.3f}",
        f"- **Recommended balance_skew: {bal_fit['threshold']:.3f}** (previous 0.25; "
        f"J={bal_fit['j']:.3f}, TPR={bal_fit['tpr']:.3f}, FPR={bal_fit['fpr']:.3f})",
        f"- |balance| distribution: heavy median={np.median(heavy):.3f}, "
        f"Balanced median={np.median(balanced):.3f}",
        f"- Balance-direction accuracy among directional labels (left negative, right positive): {sign_ok:.3f}",
        "",
        "Usage: update configs/default.yaml, then rerun build_evidence and C2/C3.",
        "See the README provenance note: design iteration used test diagnostics; fitting used training data only.",
    ]
    out = REPO / cfg["paths"]["reports"] / "calibration_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))

    # Frozen artifact: sorted training values for percentile context in rendering.
    calib = {
        "meta": {"source": "ShotQA sft.json train split", "n": len(rows),
                 "seed": cfg["seed"], "per_label_cap": args.per_label},
        "sym_h": sorted(round(r["sym_h"], 4) for r in rows),
        "sym_v": sorted(round(r["sym_v"], 4) for r in rows),
        "abs_balance": sorted(round(abs(r["balance"]), 4) for r in rows),
        "fitted": {"symmetry_strong": sym_fit["threshold"],
                   "balance_skew": bal_fit["threshold"],
                   "balance_sign_accuracy": float(sign_ok)},
    }
    cpath = REPO / "configs" / "calibration.json"
    cpath.write_text(json.dumps(calib), encoding="utf-8")
    print(f"calibration artifact -> {cpath}")

    # labeled raw rows for gate fitting (scripts/fit_gates.py)
    rpath = REPO / "configs" / "calibration_rows.json"
    rpath.write_text(json.dumps([
        {"label": r["label"], "sym_h": round(r["sym_h"], 4),
         "balance": round(r["balance"], 4)} for r in rows]), encoding="utf-8")
    print(f"labeled rows -> {rpath}")


if __name__ == "__main__":
    main()
