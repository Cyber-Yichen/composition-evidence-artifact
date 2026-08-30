from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

REPORTED = {
    "C0 zero-shot": ("results/eval/qwen2_5_vl_7b_C0_sc.jsonl", 479, 237),
    "CoT only": ("results/eval/qwen2_5_vl_7b_C1_sc.jsonl", 479, 227),
    "naive pre-question": ("results/eval/qwen2_5_vl_7b_C2_sc.jsonl", 479, 222),
    "Channel A": (
        "results/ablation/qwen2_5_vl_7b_gatedJ_after_sc.jsonl", 479, 269),
    "Channel B": ("results/eval/qwen2_5_vl_7b_C4_sc.jsonl", 479, 292),
    "Channel B, eight tokens": (
        "results/eval/qwen2_5_vl_7b_C4k8_sc.jsonl", 479, 288),
    "rule baseline": ("results/eval/rule_baseline_RULE_sc.jsonl", 479, 181),
    "naive, all fields corrupted": (
        "results/ablation/qwen2_5_vl_7b_perturb_all_sc.jsonl", 479, 161),
    "Channel A, all fields corrupted": (
        "results/ablation/qwen2_5_vl_7b_pfinal_all_after_sc.jsonl", 479, 232),
    "Channel A, position corrupted": (
        "results/ablation/qwen2_5_vl_7b_pfinal_position_after_sc.jsonl",
        479,
        259,
    ),
    "Channel A, mass corrupted": (
        "results/ablation/qwen2_5_vl_7b_pfinal_balance_after_sc.jsonl",
        479,
        245,
    ),
    "Channel A, mirror corrupted": (
        "results/ablation/qwen2_5_vl_7b_pfinal_symmetry_after_sc.jsonl",
        479,
        270,
    ),
    "ungated post-question": (
        "results/ablation/qwen2_5_vl_7b_pos_after_sc.jsonl", 479, 256),
    "strict-gated post-question": (
        "results/ablation/qwen2_5_vl_7b_gated_after_sc.jsonl", 479, 261),
    "Qwen2.5-VL-7B + Channel A": (
        "results/ablation/qwen2_5_vl_7b_gatedJ_after_sc.jsonl", 479, 269),
    "InternVL3-8B C0": (
        "results/eval/internvl3_8b_C0_sc.jsonl", 479, 218),
    "InternVL3-8B + Channel A": (
        "results/eval/internvl3_8b_FINAL_sc.jsonl", 479, 229),
    "ShotVL-7B C0": ("results/eval/shotvl_7b_C0_sc.jsonl", 479, 298),
    "ShotVL-7B + Channel A": (
        "results/eval/shotvl_7b_FINAL_sc.jsonl", 479, 305),
    "GPT-5.4 C0": ("results/eval/gpt_5_4_C0_sc.jsonl", 479, 283),
    "GPT-5.4 + Channel A": (
        "results/eval/gpt_5_4_FINAL_sc.jsonl", 479, 293),
    "Gemini-3.1-Pro C0": (
        "results/eval/gemini_3_1_pro_C0_sc.jsonl", 479, 284),
    "Gemini-3.1-Pro + Channel A": (
        "results/eval/gemini_3_1_pro_FINAL_sc.jsonl", 479, 290),
    "Qwen3-VL-Plus C0": (
        "results/eval/qwen3_vl_plus_C0_sc.jsonl", 479, 238),
    "Qwen3-VL-Plus + Channel A": (
        "results/eval/qwen3_vl_plus_FINAL_sc.jsonl", 479, 239),
    "Qwen-VL-Max C0": (
        "results/eval/qwen_vl_max_C0_sc.jsonl", 479, 257),
    "Qwen-VL-Max + Channel A": (
        "results/eval/qwen_vl_max_FINAL_sc.jsonl", 479, 253),
    "LLaVA-OneVision-7B C0": (
        "results/eval/llava_onevision_7b_C0_sc.jsonl", 479, 210),
    "LLaVA-OneVision-7B + Channel A": (
        "results/eval/llava_onevision_7b_FINAL_sc.jsonl", 479, 189),
    "Claude-Opus-4.8 C0": (
        "results/eval/claude_opus_4_8_C0_sc.jsonl", 479, 302),
    "Claude-Opus-4.8 + Channel A": (
        "results/eval/claude_opus_4_8_FINAL_sc.jsonl", 479, 269),
    "ShotBench other seven, C0": (
        "results/eval/qwen2_5_vl_7b_C0_other7.jsonl", 3093, 1637),
    "ShotBench other seven, Channel A": (
        "results/eval/qwen2_5_vl_7b_FINAL_other7.jsonl", 3093, 1745),
    "CineTechBench C0": (
        "results/eval/ctb_qwen2_5_vl_7b_C0.jsonl", 120, 78),
    "CineTechBench Channel A": (
        "results/eval/ctb_qwen2_5_vl_7b_FINAL.jsonl", 120, 77),
}

PAIRED = {
    "naive vs C0": (
        "results/eval/qwen2_5_vl_7b_C0_sc.jsonl",
        "results/eval/qwen2_5_vl_7b_C2_sc.jsonl",
        75,
        90,
    ),
    "Channel A vs C0": (
        "results/eval/qwen2_5_vl_7b_C0_sc.jsonl",
        "results/ablation/qwen2_5_vl_7b_gatedJ_after_sc.jsonl",
        66,
        34,
    ),
    "Channel B vs C0": (
        "results/eval/qwen2_5_vl_7b_C0_sc.jsonl",
        "results/eval/qwen2_5_vl_7b_C4_sc.jsonl",
        100,
        45,
    ),
    "Channel B vs Channel A": (
        "results/ablation/qwen2_5_vl_7b_gatedJ_after_sc.jsonl",
        "results/eval/qwen2_5_vl_7b_C4_sc.jsonl",
        79,
        56,
    ),
}


def read_records(relative_path: str) -> list[dict]:
    path = REPO / relative_path
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


def check_reported_rows() -> None:
    print("Reported result records")
    print("Condition                                      Correct/N   Accuracy")
    for label, (path, expected_n, expected_correct) in REPORTED.items():
        records = read_records(path)
        n = len(records)
        correct = sum(bool(row["correct"]) for row in records)
        if (n, correct) != (expected_n, expected_correct):
            raise AssertionError(
                f"{label}: expected {expected_correct}/{expected_n}, "
                f"found {correct}/{n}")
        print(f"{label:46s} {correct:4d}/{n:<4d}   {correct / n:7.1%}")


def check_paired_rows() -> None:
    print("\nPaired transition checks")
    print("Comparison                              Gained/Lost   Exact McNemar p")
    for label, (base_path, new_path, expected_gained, expected_lost) in PAIRED.items():
        base = {
            row["uid"]: bool(row["correct"])
            for row in read_records(base_path)
        }
        new = {
            row["uid"]: bool(row["correct"])
            for row in read_records(new_path)
        }
        if set(base) != set(new):
            raise AssertionError(f"{label}: result files do not share identical UIDs")
        gained = sum(not base[uid] and new[uid] for uid in base)
        lost = sum(base[uid] and not new[uid] for uid in base)
        if (gained, lost) != (expected_gained, expected_lost):
            raise AssertionError(
                f"{label}: expected {expected_gained}/{expected_lost}, "
                f"found {gained}/{lost}")
        print(
            f"{label:39s} {gained:3d}/{lost:<3d}       "
            f"{exact_mcnemar(gained, lost):.6g}")


def check_threshold_sweep() -> None:
    paths = sorted(
        (REPO / "results" / "ablation").glob(
            "qwen2_5_vl_7b_thr_ct*_sy*_after_sc.jsonl"))
    if len(paths) != 9:
        raise AssertionError(f"threshold sweep: expected 9 files, found {len(paths)}")
    accuracies = []
    for path in paths:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != 479:
            raise AssertionError(f"{path.name}: expected 479 records")
        accuracies.append(sum(bool(row["correct"]) for row in records) / len(records))
    low, high = min(accuracies), max(accuracies)
    if round(low * 100, 1) != 56.8 or round(high * 100, 1) != 57.0:
        raise AssertionError(
            f"threshold sweep: expected 56.8%-57.0%, found "
            f"{low:.1%}-{high:.1%}")
    print(f"\nThreshold sweep: {len(paths)} settings, {low:.1%}-{high:.1%}")


def main() -> None:
    check_reported_rows()
    check_paired_rows()
    check_threshold_sweep()
    print("\nAll checked counts match the shipped per-example records.")


if __name__ == "__main__":
    main()
