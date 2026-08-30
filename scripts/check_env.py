from __future__ import annotations

import argparse
import importlib
import platform
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.utils import REPO_ROOT, load_config  # noqa: E402

PACKAGES = ["numpy", "cv2", "skimage", "scipy", "pandas", "yaml", "matplotlib",
            "torch", "transformers", "accelerate", "qwen_vl_utils", "vllm",
            "openai", "google.genai", "anthropic"]


def synthetic_image() -> np.ndarray:
    img = np.zeros((240, 360, 3), dtype=np.uint8)
    img[60:180, 200:280] = 200
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    lines = [f"# Environment check ({date.today()})", "",
             f"- python: {sys.version.split()[0]}  platform: {platform.platform()}"]

    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            lines.append(f"- {name}: {getattr(mod, '__version__', 'ok')}")
        except ImportError:
            lines.append(f"- {name}: **MISSING**")

    try:
        import torch

        lines.append(f"- CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            lines.append(f"- GPU: {p.name}, {p.total_memory / 1e9:.1f} GB")
    except ImportError:
        lines.append("- CUDA: torch is not installed (optional for CPU-only checks)")

    # CPU extractor smoke: lines + saliency + full feature pass on a synthetic image
    try:
        import cv2

        from src.extractors.lines import detect_segments
        from src.features.compute import compute_features

        img = synthetic_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        segs = detect_segments(gray, cfg["extractors"]["lines"]["detector"])
        feats = compute_features(
            {"boxes": [{"box": [0.55, 0.25, 0.78, 0.75], "score": 0.9,
                        "phrase": "person"}],
             "gray": gray, "closeness": None, "segments": segs}, cfg)
        lines.append(f"- CPU smoke: {len(segs)} segments; features computed "
                     f"(no_subject={feats['no_subject']}) OK")
    except Exception as e:  # noqa: BLE001 — smoke test reports, never crashes
        lines.append(f"- CPU smoke: **FAILED** {type(e).__name__}: {e}")

    if args.full:
        cache = REPO_ROOT / cfg["paths"]["cache"]
        for label, build in [
            ("grounding_dino", lambda: _smoke_dino(cache, cfg)),
            ("depth", lambda: _smoke_depth(cache, cfg)),
        ]:
            try:
                build()
                lines.append(f"- {label} smoke: OK")
            except Exception as e:  # noqa: BLE001
                lines.append(f"- {label} smoke: **FAILED** {type(e).__name__}: {e}")

    out = REPO_ROOT / cfg["paths"]["reports"] / "environment_check.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten -> {out}")


def _smoke_dino(cache, cfg):
    from src.extractors.grounding_dino import GroundingDINOExtractor

    g = cfg["extractors"]["grounding_dino"]
    ex = GroundingDINOExtractor(cache, g["model_id"], g["prompts"],
                                g["box_threshold"], g["text_threshold"])
    ex.extract(synthetic_image())


def _smoke_depth(cache, cfg):
    from src.extractors.depth import DepthExtractor

    d = cfg["extractors"]["depth"]
    ex = DepthExtractor(cache, d["model_id"], d["larger_is_closer"])
    ex.extract(synthetic_image())


if __name__ == "__main__":
    main()
