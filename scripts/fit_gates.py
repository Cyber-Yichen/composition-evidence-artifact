from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


def fit_balance_gate(rows: list[dict], precision_target: float) -> dict:
    """Smallest |balance| threshold whose train precision (heavy & correct
    direction) meets the target."""
    best = None
    for tau in np.arange(0.05, 0.61, 0.01):
        sel = [r for r in rows if abs(r["balance"]) >= tau]
        if len(sel) < 20:
            break
        hit = [r for r in sel
               if (r["label"] == "Left heavy" and r["balance"] < 0)
               or (r["label"] == "Right heavy" and r["balance"] > 0)]
        prec = len(hit) / len(sel)
        cov = len(sel) / len(rows)
        if prec >= precision_target:
            best = {"balance_min": round(float(tau), 3),
                    "precision": round(prec, 3), "coverage": round(cov, 3)}
            break
    if best is None:  # target unreachable -> most precise point with n>=20
        taus = [(tau, [r for r in rows if abs(r["balance"]) >= tau])
                for tau in np.arange(0.05, 0.61, 0.01)]
        taus = [(t, s) for t, s in taus if len(s) >= 20]
        t, sel = max(taus, key=lambda ts: sum(
            (r["label"] == "Left heavy" and r["balance"] < 0)
            or (r["label"] == "Right heavy" and r["balance"] > 0)
            for r in ts[1]) / len(ts[1]))
        hit = sum((r["label"] == "Left heavy" and r["balance"] < 0)
                  or (r["label"] == "Right heavy" and r["balance"] > 0)
                  for r in sel)
        best = {"balance_min": round(float(t), 3),
                "precision": round(hit / len(sel), 3),
                "coverage": round(len(sel) / len(rows), 3),
                "note": "precision target unreachable; max-precision point used"}
    return best


def fit_sym_gate(rows: list[dict], coverage_min: float) -> dict:
    """sym_hi = threshold maximizing precision for GT=Symmetrical subject to
    coverage >= coverage_min; sym_lo = P10 of sym_h."""
    vals = sorted(r["sym_h"] for r in rows)
    best = {"sym_hi": vals[-1], "precision": 0.0, "coverage": 0.0}
    for tau in np.arange(0.2, 0.81, 0.01):
        sel = [r for r in rows if r["sym_h"] >= tau]
        cov = len(sel) / len(rows)
        if cov < coverage_min or not sel:
            continue
        prec = sum(r["label"] == "Symmetrical" for r in sel) / len(sel)
        if prec > best["precision"]:
            best = {"sym_hi": round(float(tau), 3), "precision": round(prec, 3),
                    "coverage": round(cov, 3)}
    base = sum(r["label"] == "Symmetrical" for r in rows) / len(rows)
    best["base_rate"] = round(base, 3)
    best["lift"] = round(best["precision"] / base, 2) if base else None
    best["sym_lo"] = round(float(np.percentile([r["sym_h"] for r in rows], 10)), 3)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bal-precision", type=float, default=0.70)
    ap.add_argument("--sym-coverage", type=float, default=0.08)
    args = ap.parse_args()

    rows = json.loads((REPO / "configs" / "calibration_rows.json").read_text())
    bal = fit_balance_gate(rows, args.bal_precision)
    sym = fit_sym_gate(rows, args.sym_coverage)

    cpath = REPO / "configs" / "calibration.json"
    calib = json.loads(cpath.read_text())
    calib["gates"] = {
        "balance_min": bal["balance_min"],
        "sym_hi": sym["sym_hi"],
        "sym_lo": sym["sym_lo"],
        "edge_only_when_near": True,
        "fit": {"balance": bal, "symmetry": sym,
                "n_train": len(rows), "bal_precision_target": args.bal_precision,
                "sym_coverage_min": args.sym_coverage},
    }
    cpath.write_text(json.dumps(calib), encoding="utf-8")

    rep = ["# Evidence-gate fitting report (ShotQA training split)", "",
           f"- Training rows: {len(rows)}",
           f"- **balance_min = {bal['balance_min']}** (direction precision {bal['precision']}, "
           f"coverage {bal['coverage']}{'; ' + bal.get('note', '') if bal.get('note') else ''})",
           f"- **sym_hi = {sym['sym_hi']}** (precision {sym['precision']} vs base rate "
           f"{sym['base_rate']}, lift {sym['lift']}x, coverage {sym['coverage']})",
           f"- **sym_lo = {sym['sym_lo']}** (P10; lower values indicate low mirror similarity)",
           "- The edge line is emitted only for near-edge or full-span flags",
           "", "Written to configs/calibration.json[gates] and consumed by render_evidence(gates=...)."]
    out = REPO / "reports" / "gates_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rep) + "\n", encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
