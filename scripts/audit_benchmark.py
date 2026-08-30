from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.eval.datasets import SC_CATEGORY, load_shotbench  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    root = REPO / cfg["paths"]["shotbench"]
    samples = load_shotbench(root)
    df = pd.DataFrame([{
        "uid": s.uid, "category": s.category, "media_type": s.media_type,
        "n_options": len(s.options), "answer": s.answer,
        "answer_text": s.options[s.answer], "media_path": s.media_paths[0],
    } for s in samples])

    lines = [f"# ShotBench metadata audit ({date.today()})", ""]
    lines.append(f"total samples: **{len(df)}**")
    lines.append("\n## Samples per category (dimension)\n")
    lines += [f"- {k}: {v}" for k, v in df.category.value_counts().items()]

    sc = df[df.category == SC_CATEGORY]
    lines.append(f"\n## SC (composition) subset: {len(sc)} samples\n")
    lines.append(f"- media types: {dict(sc.media_type.value_counts())}")
    lines.append(f"- options per question: {dict(sc.n_options.value_counts())}")

    opt_vocab: Counter = Counter()
    for s in samples:
        if s.category == SC_CATEGORY:
            opt_vocab.update(s.options.values())
    lines.append("\n### SC option vocabulary (n shown as option)\n")
    lines += [f"- {v:4d}  {k}" for k, v in opt_vocab.most_common()]
    lines.append("\n### SC ground-truth answer distribution\n")
    lines += [f"- {v:4d}  {k}" for k, v in sc.answer_text.value_counts().items()]

    # Resolution statistics are available only when media have been extracted.
    img_dir_present = any((root / p).exists() for p in sc.media_path.head(5))
    if img_dir_present:
        import cv2

        shapes = []
        for p in sc.media_path:
            img = cv2.imread(str(root / p))
            if img is not None:
                shapes.append(img.shape[:2])
        res = Counter(f"{w}x{h}" for h, w in shapes)
        lines.append("\n### SC image resolutions\n")
        lines += [f"- {k}: {v}" for k, v in res.most_common(10)]
    else:
        lines.append("\n_(Media are not present; resolution statistics were skipped.)_")

    out_md = REPO / cfg["paths"]["reports"] / "benchmark_audit.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_parquet(root / "meta.parquet", index=False)
    print(f"written -> {out_md}\nwritten -> {root / 'meta.parquet'}")

    # machine-readable label set for downstream scripts
    label_file = root / "sc_labels.json"
    label_file.write_text(json.dumps(sorted(opt_vocab), ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"written -> {label_file}")


if __name__ == "__main__":
    main()
