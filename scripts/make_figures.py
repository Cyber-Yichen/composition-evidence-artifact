from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.datasets import load_shotbench, sc_subset  # noqa: E402
from src.utils import load_config, read_jsonl  # noqa: E402
from src.viz.figures import confusion_matrix, flip_buckets, plot_confusion  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2_5_vl_7b")
    args = ap.parse_args()

    cfg = load_config()
    fig_dir = REPO / cfg["paths"]["figures"]
    fig_dir.mkdir(exist_ok=True)
    eval_dir = REPO / cfg["paths"]["results"] / "eval"

    c0 = read_jsonl(eval_dir / f"{args.model}_C0_sc.jsonl")
    c2 = read_jsonl(eval_dir / f"{args.model}_C2_sc.jsonl")
    if not c0 or not c2:
        sys.exit("need C0 and C2 results first (scripts/run_eval.py)")

    samples = sc_subset(load_shotbench(REPO / cfg["paths"]["shotbench"]))
    option_texts = {s.uid: s.options for s in samples}
    labels = sorted({t for opts in option_texts.values() for t in opts.values()
                     if "," not in t})  # base labels; combos are rare, shown as misc

    for name, recs in [("C0", c0), ("C2", c2)]:
        m = confusion_matrix(recs, labels, option_texts)
        plot_confusion(m, labels, f"{args.model} {name} (SC)",
                       fig_dir / f"confusion_{args.model}_{name}.pdf")

    # Bucket prediction flips by distance to the nearest thirds intersection.
    ev = read_jsonl(REPO / cfg["paths"]["results"] / "evidence" / "shotbench_sc.jsonl")
    d_thirds = {r["uid"]: r["features"]["subject"]["d_thirds"]
                for r in ev if r["features"].get("subject")}
    rows = flip_buckets(c0, c2, d_thirds)
    print("flip buckets (by d_thirds):")
    for r in rows:
        print(f"  {r['bucket']}  n={r['n']:4d}  C0={r['acc_base']:.3f}  "
              f"C2={r['acc_new']:.3f}")

    # candidate case-study uids: C0 wrong -> C2 right (and the reverse, shown honestly)
    base = {r["uid"]: r for r in c0}
    fixed = [u for u, r in {r["uid"]: r for r in c2}.items()
             if r["correct"] and u in base and not base[u]["correct"]]
    broken = [u for u, r in {r["uid"]: r for r in c2}.items()
              if not r["correct"] and u in base and base[u]["correct"]]
    print(f"\nC0->C2 fixed: {len(fixed)} (case-study pool), broken: {len(broken)}")
    (fig_dir / "case_uids.txt").write_text(
        "fixed:\n" + "\n".join(fixed) + "\n\nbroken:\n" + "\n".join(broken),
        encoding="utf-8")
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
