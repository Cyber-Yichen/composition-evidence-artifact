from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.datasets import load_shotbench, sc_subset  # noqa: E402
from src.injection.perturb import MODES, perturb_features  # noqa: E402
from src.injection.render import GROUPS, render_evidence  # noqa: E402
from src.utils import load_config, load_prompts, load_yaml, read_jsonl, set_seed  # noqa: E402


def load_features(cfg) -> dict[str, dict]:
    rows = read_jsonl(REPO / cfg["paths"]["results"] / "evidence" / "shotbench_sc.jsonl")
    return {r["uid"]: r["features"] for r in rows}


def rethreshold(feats: dict, thr: dict) -> dict:
    """Recompute threshold-dependent judgments without touching raw measurements."""
    f = copy.deepcopy(feats)
    s = f.get("subject")
    if s:
        s["center_aligned"] = s["d_center"] < thr["center_align"]
        s["thirds_aligned"] = s["d_thirds"] < thr["thirds_align"]
        s["edge"]["near_edge"] = s["edge"]["min_edge_dist"] < thr["edge_near"]
    if f.get("symmetry"):
        f["symmetry"]["strong_h"] = f["symmetry"]["sym_h"] > thr["symmetry_strong"]
        f["symmetry"]["strong_v"] = f["symmetry"]["sym_v"] > thr["symmetry_strong"]
    if f.get("balance"):
        b = f["balance"]["balance"]
        f["balance"]["skew"] = (None if abs(b) <= thr["balance_skew"]
                                else ("left" if b < 0 else "right"))
    return f


def evidence_variants(mode: str, cfg: dict, prompts: dict,
                      features: dict[str, dict],
                      calibration: dict | None = None) -> dict[str, dict[str, str]]:
    """Returns variant_name -> {uid -> evidence_text}."""
    thr = cfg["thresholds"]
    out: dict[str, dict[str, str]] = {}

    def rend(f: dict, **kw) -> str:
        return render_evidence(f, prompts, calibration=calibration, **kw)

    if mode == "groups":
        for gname, gset in GROUPS.items():
            out[f"only_{gname}"] = {u: rend(f, include=gset)
                                    for u, f in features.items()}
        all_f = {f for g in GROUPS.values() for f in g}
        for gname, gset in GROUPS.items():
            keep = all_f - gset
            out[f"loo_{gname}"] = {u: rend(f, include=keep)
                                   for u, f in features.items()}

    elif mode == "perturb":
        for pmode in MODES:
            out[f"perturb_{pmode}"] = {
                u: rend(perturb_features(f, pmode, thr, seed=cfg["seed"]))
                for u, f in features.items()}

    elif mode == "thresholds":
        # sensitivity grid under the FINAL recipe (gates + after-position);
        # symmetry_strong doubles as the sym_hi gate in the final recipe
        grid = cfg["threshold_grid"]
        for ct in grid["center_thirds"]:
            for sy in grid["symmetry"]:
                t = {**thr, "center_align": ct, "thirds_align": ct,
                     "symmetry_strong": sy}
                gates_t = {"balance_min": t["balance_skew"], "sym_hi": sy,
                           "sym_lo": (calibration or {}).get("gates", {}).get(
                               "sym_lo", 0.0),
                           "edge_only_when_near": True}
                calib_t = dict(calibration or {}, gates=gates_t)
                out[f"thr_ct{ct}_sy{sy}_after"] = {
                    u: render_evidence(rethreshold(f, t), prompts,
                                       calibration=calib_t, use_gates=True)
                    for u, f in features.items()}

    elif mode == "position":
        out["pos_after"] = {u: rend(f) for u, f in features.items()}

    elif mode == "style":
        out["style_numeric"] = {u: rend(f, style="numeric")
                                for u, f in features.items()}

    elif mode == "gated":
        if not (calibration or {}).get("gates"):
            raise SystemExit("no gates in configs/calibration.json — "
                             "run scripts/fit_gates.py first")
        out["gated"] = {u: rend(f, use_gates=True) for u, f in features.items()}

    elif mode == "perturb_final":
        # robustness of the FINAL recipe: perturbed features, Youden gates, after
        gates_j = {"balance_min": thr["balance_skew"],
                   "sym_hi": thr["symmetry_strong"],
                   "sym_lo": (calibration or {}).get("gates", {}).get("sym_lo", 0.0),
                   "edge_only_when_near": True}
        calib_j = dict(calibration or {}, gates=gates_j)
        for pmode in MODES:
            out[f"pfinal_{pmode}_after"] = {
                u: render_evidence(perturb_features(f, pmode, thr, seed=cfg["seed"]),
                                   prompts, calibration=calib_j, use_gates=True)
                for u, f in features.items()}

    elif mode == "after_gates":
        # Evidence after the question with two training-fitted gate strengths:
        # strict precision gates (fit_gates) and Youden thresholds (config)
        if not (calibration or {}).get("gates"):
            raise SystemExit("run scripts/fit_gates.py first")
        out["gated_after"] = {u: rend(f, use_gates=True)
                              for u, f in features.items()}
        gates_j = {"balance_min": thr["balance_skew"],
                   "sym_hi": thr["symmetry_strong"],
                   "sym_lo": calibration["gates"]["sym_lo"],
                   "edge_only_when_near": True}
        calib_j = dict(calibration, gates=gates_j)
        out["gatedJ_after"] = {
            u: render_evidence(f, prompts, calibration=calib_j, use_gates=True)
            for u, f in features.items()}
    else:
        raise ValueError(mode)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["groups", "perturb", "thresholds", "position", "style",
                             "gated", "after_gates", "perturb_final"])
    ap.add_argument("--model", default="qwen2_5_vl_7b")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    prompts = load_prompts()
    set_seed(cfg["seed"])
    features = load_features(cfg)
    if not features:
        sys.exit("no cached features — run scripts/build_evidence.py first")

    import json as _json
    calib_path = REPO / "configs" / "calibration.json"
    calibration = (_json.loads(calib_path.read_text()) if calib_path.exists()
                   else None)
    variants = evidence_variants(args.mode, cfg, prompts, features, calibration)

    models = load_yaml(REPO / "configs" / "models.yaml")
    mcfg = models["local"].get(args.model) or models["api"].get(args.model)
    root = REPO / cfg["paths"]["shotbench"]
    samples = sc_subset(load_shotbench(root))
    if args.limit:
        samples = samples[:args.limit]

    from src.eval.backends import build_backend
    from src.eval.runner import run_condition

    backend = build_backend(args.model, mcfg, cfg["eval"])
    out_dir = REPO / cfg["paths"]["results"] / "ablation"
    for vname, ev in variants.items():
        out = out_dir / f"{args.model}_{vname}_sc.jsonl"
        position = ("after_question"
                    if vname == "pos_after" or vname.endswith("_after") else None)
        run_condition(samples, backend, "C2", out, args.model, cfg, prompts,
                      evidence_texts=ev, bench_root=root, position=position)
        rows = read_jsonl(out)
        acc = sum(r["correct"] for r in rows) / len(rows)
        print(f"{vname}: acc={acc:.4f} (n={len(rows)})")
        from src.eval.tblog import log_result

        log_result(f"ablation_{args.model}_{vname}", {"accuracy": acc}, cfg)


if __name__ == "__main__":
    main()
