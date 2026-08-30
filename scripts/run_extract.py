from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
from rich.progress import track  # noqa: E402

from src.eval.datasets import load_shotbench, sc_subset  # noqa: E402
from src.eval.runner import media_for_sample  # noqa: E402
from src.extractors.depth import DepthExtractor  # noqa: E402
from src.extractors.grounding_dino import GroundingDINOExtractor  # noqa: E402
from src.extractors.lines import LineExtractor  # noqa: E402
from src.utils import file_key, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-categories", action="store_true",
                    help="also extract for the other 7 dims (harmlessness eval)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    root = REPO / cfg["paths"]["shotbench"]
    cache = REPO / cfg["paths"]["cache"]
    samples = load_shotbench(root)
    if not args.all_categories:
        samples = sc_subset(samples)
    if args.limit:
        samples = samples[:args.limit]

    g = cfg["extractors"]["grounding_dino"]
    d = cfg["extractors"]["depth"]
    dino = GroundingDINOExtractor(cache, g["model_id"], g["prompts"],
                                  g["box_threshold"], g["text_threshold"])
    depth = DepthExtractor(cache, d["model_id"], d["larger_is_closer"])
    lines = LineExtractor(cache, cfg["extractors"]["lines"]["detector"])

    stats = {"n": 0, "no_boxes": 0, "total_boxes": 0, "no_segments": 0}
    for s in track(samples, description="extracting"):
        # Composition features for videos use the middle sampled frame.
        media = media_for_sample(s, root, cfg["video"]["n_frames"] - 1)
        target = media[len(media) // 2] if s.media_type == "video" else media[0]
        img = cv2.imread(str(target))
        if img is None:
            print(f"[WARN] unreadable media for {s.uid}: {target}")
            continue
        key = file_key(target)
        boxes = dino(img, key)["boxes"]
        depth(img, key)
        segs = lines(img, key)["segments"]
        stats["n"] += 1
        stats["total_boxes"] += len(boxes)
        stats["no_boxes"] += not boxes
        stats["no_segments"] += not segs

    rep = [f"# Extractor coverage ({date.today()})", "",
           f"- samples processed: {stats['n']}",
           f"- avg boxes/sample: {stats['total_boxes'] / max(1, stats['n']):.2f}",
           f"- no-subject samples: {stats['no_boxes']} "
           f"({stats['no_boxes'] / max(1, stats['n']):.1%})",
           f"- no-segment samples: {stats['no_segments']}"]
    out = REPO / cfg["paths"]["reports"] / "extractor_coverage.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
