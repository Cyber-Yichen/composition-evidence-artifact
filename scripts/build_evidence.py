from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rich.progress import track  # noqa: E402

from src.eval.datasets import load_shotbench, sc_subset  # noqa: E402
from src.eval.runner import media_for_sample  # noqa: E402
from src.features.compute import compute_features  # noqa: E402
from src.injection.render import render_evidence  # noqa: E402
from src.utils import (append_jsonl, done_uids, file_key,  # noqa: E402
                       load_config, load_prompts, set_seed)


def load_cached(cache: Path, key: str) -> dict | None:
    import json

    dino_f = cache / "grounding_dino" / f"{key}.json"
    depth_f = cache / "depth" / f"{key}.npz"
    lines_f = cache / "lines" / f"{key}.json"
    if not dino_f.exists() or not lines_f.exists():
        return None
    out = {
        "boxes": json.loads(dino_f.read_text())["boxes"],
        "segments": json.loads(lines_f.read_text())["segments"],
        "closeness": None,
    }
    if depth_f.exists():
        with np.load(depth_f) as z:
            out["closeness"] = z["closeness"].astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-categories", action="store_true")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    prompts = load_prompts()
    set_seed(cfg["seed"])
    calib_path = REPO / "configs" / "calibration.json"
    calibration = (json.loads(calib_path.read_text()) if calib_path.exists()
                   else None)
    if calibration:
        print(f"using calibration artifact: n={calibration['meta']['n']}")
    root = REPO / cfg["paths"]["shotbench"]
    cache = REPO / cfg["paths"]["cache"]
    subset = "all" if args.all_categories else "sc"
    out_path = REPO / cfg["paths"]["results"] / "evidence" / f"shotbench_{subset}.jsonl"

    samples = load_shotbench(root)
    if not args.all_categories:
        samples = sc_subset(samples)

    done = done_uids(out_path)
    missing_cache = 0
    for s in track(samples, description="building evidence"):
        if s.uid in done:
            continue
        media = media_for_sample(s, root, cfg["video"]["n_frames"] - 1)
        target = media[len(media) // 2] if s.media_type == "video" else media[0]
        key = file_key(target)
        cached = load_cached(cache, key)
        if cached is None:
            missing_cache += 1
            continue
        img = cv2.imread(str(target))
        cached["gray"] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = compute_features(cached, cfg)
        text = render_evidence(feats, prompts,
                               guard=cfg["evidence"]["lexicon_guard"],
                               calibration=calibration)
        append_jsonl(out_path, {"uid": s.uid, "features": feats,
                                "evidence_text": text})
    if missing_cache:
        print(f"[WARN] {missing_cache} samples had no extractor cache — "
              "run scripts/run_extract.py first")
    print(f"evidence -> {out_path}")

    if args.sample:
        from src.utils import read_jsonl

        rows = read_jsonl(out_path)[: args.sample]
        md = ["# Evidence text samples for qualitative inspection", ""]
        for r in rows:
            md += [f"## {r['uid']}", "```", r["evidence_text"], "```", ""]
        p = REPO / cfg["paths"]["reports"] / "evidence_samples.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(md), encoding="utf-8")
        print(f"samples -> {p}")


if __name__ == "__main__":
    main()
