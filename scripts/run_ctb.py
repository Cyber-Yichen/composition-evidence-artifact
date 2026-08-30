from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rich.progress import track  # noqa: E402

from src.extractors.depth import DepthExtractor  # noqa: E402
from src.extractors.grounding_dino import GroundingDINOExtractor  # noqa: E402
from src.extractors.lines import LineExtractor  # noqa: E402
from src.features.compute import compute_features  # noqa: E402
from src.injection.render import lexicon_guard, render_evidence  # noqa: E402
from src.utils import (append_jsonl, done_uids, file_key, load_config,  # noqa: E402
                       load_prompts, read_jsonl, set_seed)

CTB_EXTRA_BANNED = ["rule of thirds", "thirds-aligned", "framing", "diagonal",
                    "curved", "horizontal", "central"]


def load_ctb(root: Path) -> list[dict]:
    ann = json.load(open(root / "image_annotation.json", encoding="utf-8"))
    items = []
    for x in ann:
        if not x["Category"].lower().startswith("comp"):
            continue
        prompt = x["QA"]["Prompt"]
        opts = dict(re.findall(r"([A-J])\.\s*([^\n]+)", prompt))
        items.append({"uid": f"ctb-{x['Image_name']}", "image": x["Image_name"],
                      "prompt": prompt, "options": {k: v.strip() for k, v in
                                                    opts.items()},
                      "answer": x["QA"]["Answer"].strip(),
                      "label": x["Annotation"]})
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    prompts = load_prompts()
    set_seed(cfg["seed"])
    root = REPO / "data" / "cinetechbench"
    items = load_ctb(root)
    if args.limit:
        items = items[: args.limit]
    print(f"CTB composition items: {len(items)}")

    # CTB-safe template overrides (label-root suppression for THIS benchmark)
    prompts_ctb = copy.deepcopy(prompts)
    ev = prompts_ctb["evidence"]
    ev["align_thirds"] = ev["align_none"]
    ev["horizon_part"] = "; longest near-level line at y={horizon_y:.2f}"

    calib = json.loads((REPO / "configs" / "calibration.json").read_text())
    gates_j = {"balance_min": cfg["thresholds"]["balance_skew"],
               "sym_hi": cfg["thresholds"]["symmetry_strong"],
               "sym_lo": calib["gates"]["sym_lo"],
               "edge_only_when_near": True}
    calib_j = dict(calib, gates=gates_j)

    g = cfg["extractors"]["grounding_dino"]
    d = cfg["extractors"]["depth"]
    cache = REPO / cfg["paths"]["cache"]
    dino = GroundingDINOExtractor(cache, g["model_id"], g["prompts"],
                                  g["box_threshold"], g["text_threshold"])
    depth = DepthExtractor(cache, d["model_id"], d["larger_is_closer"])
    lines = LineExtractor(cache, cfg["extractors"]["lines"]["detector"])

    evidence: dict[str, str] = {}
    for it in track(items, description="ctb evidence"):
        img = cv2.imread(str(root / "images" / it["image"]))
        if img is None:
            continue
        key = file_key(root / "images" / it["image"])
        extraction = {
            "boxes": dino(img, key)["boxes"],
            "closeness": depth(img, key)["closeness"].astype(np.float32),
            "segments": lines(img, key)["segments"],
            "gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        }
        feats = compute_features(extraction, cfg)
        text = render_evidence(feats, prompts_ctb, calibration=calib_j,
                               use_gates=True, guard=True)
        lexicon_guard(text, CTB_EXTRA_BANNED)  # CTB label roots, hard check
        evidence[it["uid"]] = text

    from src.eval.backends import QwenVLBackend
    from src.eval.parsing import parse_choice
    from src.eval.tblog import log_result
    from src.utils import load_yaml

    models = load_yaml(REPO / "configs" / "models.yaml")
    backend = QwenVLBackend(models["local"]["qwen2_5_vl_7b"]["hf_id"],
                            max_new_tokens=cfg["eval"]["max_new_tokens"])

    for cond in ("C0", "FINAL"):
        out = REPO / cfg["paths"]["results"] / "eval" / f"ctb_qwen2_5_vl_7b_{cond}.jsonl"
        done = done_uids(out)
        for it in track(items, description=f"ctb {cond}"):
            if it["uid"] in done or it["uid"] not in evidence:
                continue
            prompt = it["prompt"]
            if cond == "FINAL":
                prompt = f"{prompt}\n\n{evidence[it['uid']]}"
            resp = backend.generate([root / "images" / it["image"]], prompt)
            letter, method = parse_choice(resp, it["options"])
            append_jsonl(out, {"uid": it["uid"], "model": "qwen2_5_vl_7b",
                               "condition": cond, "category": "ctb-composition",
                               "label": it["label"], "response": resp,
                               "parsed": letter, "parse_method": method,
                               "answer": it["answer"],
                               "correct": letter == it["answer"]})
        rows = read_jsonl(out)
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"CTB {cond}: acc={acc:.4f} (n={len(rows)})")
        log_result(f"ctb_qwen2_5_vl_7b_{cond}", {"accuracy": acc}, cfg)

    # per-label table
    from collections import defaultdict

    for cond in ("C0", "FINAL"):
        rows = read_jsonl(REPO / cfg["paths"]["results"] / "eval" /
                          f"ctb_qwen2_5_vl_7b_{cond}.jsonl")
        byl = defaultdict(lambda: [0, 0])
        for r in rows:
            byl[r["label"]][0] += 1
            byl[r["label"]][1] += r["correct"]
        print(f"-- {cond} per-label:")
        for l, (n, k) in sorted(byl.items(), key=lambda kv: -kv[1][0]):
            print(f"   {l:16s} n={n:3d} acc={k / n:.3f}")


if __name__ == "__main__":
    main()
