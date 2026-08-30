from __future__ import annotations

import argparse
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
from src.injection.soft_token import featurize  # noqa: E402
from src.utils import append_jsonl, done_uids, file_key, load_config, set_seed  # noqa: E402

COMP_QUESTION = "What's the composition of this shot?"


def load_comp_items(root: Path) -> list[dict]:
    """All composition turns (single-label AND combo — training uses everything)."""
    data = json.load(open(root / "sft.json", encoding="utf-8"))
    out = []
    for idx, item in enumerate(data):
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
                if letter in opts:
                    out.append({"uid": f"shotqa-{idx}", "image": imgs[0],
                                "options": {k: v.strip() for k, v in opts.items()},
                                "answer": letter})
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    root = REPO / cfg["paths"]["shotqa"]
    out_path = REPO / cfg["paths"]["results"] / "shotqa_features.jsonl"

    items = load_comp_items(root)
    print(f"composition items: {len(items)}")
    if args.limit:
        items = items[: args.limit]

    g = cfg["extractors"]["grounding_dino"]
    d = cfg["extractors"]["depth"]
    cache = REPO / cfg["paths"]["cache"]
    dino = GroundingDINOExtractor(cache, g["model_id"], g["prompts"],
                                  g["box_threshold"], g["text_threshold"])
    depth = DepthExtractor(cache, d["model_id"], d["larger_is_closer"])
    lines = LineExtractor(cache, cfg["extractors"]["lines"]["detector"])

    done = done_uids(out_path)
    skipped = 0
    for it in track(items, description="shotqa features"):
        if it["uid"] in done:
            continue
        p = root / it["image"]
        if not p.exists():
            p = root / Path(it["image"]).name
        img = cv2.imread(str(p))
        if img is None:
            skipped += 1
            continue
        key = file_key(p)
        extraction = {
            "boxes": dino(img, key)["boxes"],
            "closeness": depth(img, key)["closeness"].astype(np.float32),
            "segments": lines(img, key)["segments"],
            "gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        }
        feats = compute_features(extraction, cfg)
        append_jsonl(out_path, {"uid": it["uid"], "image": it["image"],
                                "options": it["options"], "answer": it["answer"],
                                "vec": [float(x) for x in featurize(feats)]})
    print(f"done -> {out_path} (unreadable skipped: {skipped})")


if __name__ == "__main__":
    main()
