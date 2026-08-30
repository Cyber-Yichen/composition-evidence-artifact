from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.datasets import SC_CATEGORY, load_shotbench, sc_subset  # noqa: E402
from src.utils import load_config, load_prompts, load_yaml, read_jsonl, set_seed  # noqa: E402


def load_evidence_texts(cfg) -> dict[str, str]:
    rows = []
    for name in ("shotbench_sc.jsonl", "shotbench_all.jsonl"):
        path = REPO / cfg["paths"]["results"] / "evidence" / name
        if path.exists():
            rows += read_jsonl(path)
    return {r["uid"]: r["evidence_text"] for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2_5_vl_7b")
    ap.add_argument("--conditions", default="C0")
    ap.add_argument("--subset", default="sc", choices=["sc", "other7", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    prompts = load_prompts()
    set_seed(cfg["seed"])
    out_dir = REPO / cfg["paths"]["results"] / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize(cfg, out_dir)
        return

    models = load_yaml(REPO / "configs" / "models.yaml")
    mcfg = models["local"].get(args.model) or models["api"].get(args.model)
    if mcfg is None:
        sys.exit(f"unknown model {args.model} (see configs/models.yaml)")

    root = REPO / cfg["paths"]["shotbench"]
    samples = load_shotbench(root)
    if args.subset == "sc":
        samples = sc_subset(samples)
    elif args.subset == "other7":
        samples = [s for s in samples if s.category != SC_CATEGORY]
    if args.limit:
        samples = samples[:args.limit]

    conditions = args.conditions.split(",")
    evidence = load_evidence_texts(cfg) if any(c in ("C2", "C3")
                                               for c in conditions) else None

    from src.eval.backends import build_backend  # heavy import deferred
    from src.eval.runner import run_condition

    backend = build_backend(args.model, mcfg, cfg["eval"])
    for cond in conditions:
        if cond == "C4":
            print("C4 (soft tokens) is evaluated by scripts/eval_c4.py; skipping.")
            continue
        # CoT conditions need a larger generation budget than letter answers
        budget = (cfg["eval"]["max_new_tokens_cot"] if cond in ("C1", "C3")
                  else cfg["eval"]["max_new_tokens"])
        for attr in ("max_new_tokens", "max_tokens"):
            if hasattr(backend, attr):
                setattr(backend, attr, budget)
        out = out_dir / f"{args.model}_{cond}_{args.subset}.jsonl"
        run_condition(samples, backend, cond, out, args.model, cfg, prompts,
                      evidence_texts=evidence, bench_root=root)
        rows = read_jsonl(out)
        fails = sum(r["parse_method"] == "fail" for r in rows)
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"{args.model} {cond} {args.subset}: acc={acc:.4f} "
              f"(n={len(rows)}, parse_fail={fails / len(rows):.2%})")
        from src.eval.tblog import log_result

        log_result(f"{args.model}_{cond}_{args.subset}",
                   {"accuracy": acc, "parse_fail_rate": fails / len(rows)}, cfg)
        if fails / len(rows) > cfg["eval"]["parse_fail_threshold"]:
            print("[WARN] parse failure rate above threshold — fix the PROMPT, "
                  "not the parser")


def summarize(cfg, out_dir: Path) -> None:
    import pandas as pd

    from src.eval.metrics import compare_conditions

    rows = []
    for f in sorted(out_dir.glob("*.jsonl")):
        recs = read_jsonl(f)
        if not recs:
            continue
        model, cond, subset = f.stem.rsplit("_", 2)
        rows.append({"model": model, "condition": cond, "subset": subset,
                     "n": len(recs),
                     "acc": sum(r["correct"] for r in recs) / len(recs)})
    df = pd.DataFrame(rows)
    main_csv = REPO / cfg["paths"]["results"] / "main_table.csv"
    df.to_csv(main_csv, index=False)
    print(df.to_string(index=False))

    # combo-label breakdown (labels like "Balanced, Symmetrical" are likely hard cases)
    sc_samples = {s.uid: s for s in
                  sc_subset(load_shotbench(REPO / cfg["paths"]["shotbench"]))}
    print("\nSC combo-label breakdown (answer text contains ','):")
    for f in sorted(out_dir.glob("*_sc.jsonl")):
        recs = read_jsonl(f)
        combo = [r for r in recs if r["uid"] in sc_samples
                 and "," in sc_samples[r["uid"]].options[r["answer"]]]
        base = [r for r in recs if r["uid"] in sc_samples
                and "," not in sc_samples[r["uid"]].options[r["answer"]]]
        if combo and base:
            print(f"  {f.stem}: base n={len(base)} acc="
                  f"{sum(r['correct'] for r in base) / len(base):.4f} | "
                  f"combo n={len(combo)} acc="
                  f"{sum(r['correct'] for r in combo) / len(combo):.4f}")

    # McNemar for every (model, subset): C_x vs C0
    print("\nMcNemar vs C0:")
    for (model, subset), grp in df.groupby(["model", "subset"]):
        base_f = out_dir / f"{model}_C0_{subset}.jsonl"
        if not base_f.exists():
            continue
        base = read_jsonl(base_f)
        for cond in grp.condition:
            if cond == "C0":
                continue
            new = read_jsonl(out_dir / f"{model}_{cond}_{subset}.jsonl")
            c = compare_conditions(base, new)
            print(f"  {model} {subset} {cond} vs C0: "
                  f"{c['acc_base']:.4f} -> {c['acc_new']:.4f}  "
                  f"gained={c['gained']} lost={c['lost']} p={c['p_mcnemar']:.4g}")
    print(f"\nmain table -> {main_csv}")


if __name__ == "__main__":
    main()
