from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BASELINE = REPO / "results" / "eval" / "qwen2_5_vl_7b_C0_sc.jsonl"
CHANNEL_A = (
    REPO / "results" / "ablation" / "qwen2_5_vl_7b_gatedJ_after_sc.jsonl"
)
CHANNEL_B = REPO / "results" / "eval" / "qwen2_5_vl_7b_C4_sc.jsonl"

EXPECTED = {
    "baseline": {"n": 479, "correct": 237},
    "channel_a": {"n": 479, "correct": 269, "gained": 66, "lost": 34},
    "channel_b": {"n": 479, "correct": 292, "gained": 100, "lost": 45},
}


def read_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exact_mcnemar(gained: int, lost: int) -> float:
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(gained, lost) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def summarize(name: str, records: list[dict], baseline: list[dict] | None) -> dict:
    out = {
        "n": len(records),
        "correct": sum(bool(row["correct"]) for row in records),
    }
    if baseline is not None:
        base = {row["uid"]: bool(row["correct"]) for row in baseline}
        new = {row["uid"]: bool(row["correct"]) for row in records}
        uids = set(base) & set(new)
        if len(uids) != len(base) or len(uids) != len(new):
            raise AssertionError(f"{name}: result files do not share identical UIDs")
        out["gained"] = sum(not base[uid] and new[uid] for uid in uids)
        out["lost"] = sum(base[uid] and not new[uid] for uid in uids)
        out["p_mcnemar"] = exact_mcnemar(out["gained"], out["lost"])
    return out


def main() -> None:
    baseline = read_records(BASELINE)
    rows = {
        "baseline": summarize("baseline", baseline, None),
        "channel_a": summarize("channel_a", read_records(CHANNEL_A), baseline),
        "channel_b": summarize("channel_b", read_records(CHANNEL_B), baseline),
    }
    for name, expected in EXPECTED.items():
        for key, value in expected.items():
            if rows[name][key] != value:
                raise AssertionError(
                    f"{name}.{key}: expected {value}, found {rows[name][key]}"
                )

    print("Condition   Correct/N   Accuracy   Gained/Lost vs C0   Exact McNemar p")
    for name in ("baseline", "channel_a", "channel_b"):
        row = rows[name]
        accuracy = row["correct"] / row["n"]
        flips = (
            "-"
            if name == "baseline"
            else f"{row['gained']}/{row['lost']}"
        )
        p_value = "-" if name == "baseline" else f"{row['p_mcnemar']:.6g}"
        print(
            f"{name:10s}  {row['correct']:3d}/{row['n']:<3d}   "
            f"{accuracy:8.3%}   {flips:>17s}   {p_value}"
        )
    print("All expected counts match the shipped evaluation records.")


if __name__ == "__main__":
    main()
