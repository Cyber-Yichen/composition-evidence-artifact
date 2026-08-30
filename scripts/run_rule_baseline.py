from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.datasets import load_shotbench, sc_subset  # noqa: E402
from src.eval.rule_baseline import rule_choice  # noqa: E402
from src.utils import append_jsonl, load_config, read_jsonl  # noqa: E402


def main() -> None:
    cfg = load_config()
    ev_rows = read_jsonl(REPO / cfg["paths"]["results"] / "evidence" / "shotbench_sc.jsonl")
    features = {r["uid"]: r["features"] for r in ev_rows}
    if not features:
        sys.exit("no cached features — run scripts/build_evidence.py first")

    samples = sc_subset(load_shotbench(REPO / cfg["paths"]["shotbench"]))
    out = REPO / cfg["paths"]["results"] / "eval" / "rule_baseline_RULE_sc.jsonl"
    out.unlink(missing_ok=True)  # cheap to recompute; always fresh

    n = correct = 0
    missing = 0
    for s in samples:
        f = features.get(s.uid)
        if f is None:
            missing += 1
            continue
        letter = rule_choice(f, s.options, cfg["thresholds"])
        append_jsonl(out, {"uid": s.uid, "model": "rule_baseline",
                           "condition": "RULE", "category": s.category,
                           "response": "", "parsed": letter, "parse_method": "rule",
                           "answer": s.answer, "correct": letter == s.answer})
        n += 1
        correct += letter == s.answer
    print(f"rule baseline: acc={correct / n:.4f} (n={n}, missing_features={missing})")
    print(f"-> {out}")
    from src.eval.tblog import log_result

    log_result("rule_baseline_RULE_sc", {"accuracy": correct / n}, cfg)


if __name__ == "__main__":
    main()
