from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.datasets import SC_CATEGORY, load_shotbench, sc_subset  # noqa: E402
from src.injection.render import render_evidence  # noqa: E402
from src.utils import (load_config, load_prompts, load_yaml, read_jsonl,  # noqa: E402
                       set_seed)


def gated_evidence_map(cfg, prompts) -> dict[str, str]:
    calib = json.loads((REPO / "configs" / "calibration.json").read_text())
    gates_j = {"balance_min": cfg["thresholds"]["balance_skew"],
               "sym_hi": cfg["thresholds"]["symmetry_strong"],
               "sym_lo": calib["gates"]["sym_lo"],
               "edge_only_when_near": True}
    calib_j = dict(calib, gates=gates_j)
    rows = []
    for name in ("shotbench_sc.jsonl", "shotbench_all.jsonl"):
        path = REPO / cfg["paths"]["results"] / "evidence" / name
        if path.exists():
            rows += read_jsonl(path)
    return {r["uid"]: render_evidence(r["features"], prompts,
                                      calibration=calib_j, use_gates=True)
            for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", default="C0,FINAL")
    ap.add_argument("--subset", default="sc", choices=["sc", "other7", "all"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    prompts = load_prompts()
    set_seed(cfg["seed"])
    models = load_yaml(REPO / "configs" / "models.yaml")
    mcfg = models["local"].get(args.model) or models["api"].get(args.model)
    if mcfg is None:
        sys.exit(f"unknown model {args.model}")

    root = REPO / cfg["paths"]["shotbench"]
    samples = load_shotbench(root)
    if args.subset == "sc":
        samples = sc_subset(samples)
    elif args.subset == "other7":
        samples = [s for s in samples if s.category != SC_CATEGORY]
    if args.limit:
        samples = samples[: args.limit]

    conditions = args.conditions.split(",")
    evidence = gated_evidence_map(cfg, prompts) if "FINAL" in conditions else None
    if evidence is not None:
        missing = [s.uid for s in samples if s.uid not in evidence]
        if missing:
            print(f"[WARN] {len(missing)} samples lack evidence "
                  f"(run build_evidence for this subset first); they will error")

    from src.eval.backends import build_backend
    from src.eval.runner import run_condition

    backend = build_backend(args.model, mcfg, cfg["eval"])
    out_dir = REPO / cfg["paths"]["results"] / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cond in conditions:
        out = out_dir / f"{args.model}_{cond}_{args.subset}.jsonl"
        if cond == "C0":
            run_condition(samples, backend, "C0", out, args.model, cfg, prompts,
                          bench_root=root)
        elif cond == "FINAL":
            # FINAL uses the C2 prompt structure with evidence after the question
            run_condition(samples, backend, "C2", out, args.model, cfg, prompts,
                          evidence_texts=evidence, bench_root=root,
                          position="after_question")
        else:
            sys.exit(f"run_final only handles C0/FINAL, got {cond}")
        rows = read_jsonl(out)
        acc = sum(r["correct"] for r in rows) / len(rows)
        fails = sum(r["parse_method"] == "fail" for r in rows) / len(rows)
        print(f"{args.model} {cond} {args.subset}: acc={acc:.4f} "
              f"(n={len(rows)}, parse_fail={fails:.2%})")
        from src.eval.tblog import log_result

        log_result(f"{args.model}_{cond}_{args.subset}",
                   {"accuracy": acc, "parse_fail_rate": fails}, cfg)

    # per-category table for harmlessness runs
    if args.subset == "other7":
        from collections import defaultdict

        for cond in conditions:
            rows = read_jsonl(out_dir / f"{args.model}_{cond}_{args.subset}.jsonl")
            byc = defaultdict(lambda: [0, 0])
            for r in rows:
                byc[r["category"]][0] += 1
                byc[r["category"]][1] += r["correct"]
            print(f"-- {cond} per-dimension:")
            for c, (n, k) in sorted(byc.items()):
                print(f"   {c:20s} n={n:4d} acc={k / n:.4f}")


if __name__ == "__main__":
    main()
