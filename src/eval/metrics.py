from __future__ import annotations

from scipy.stats import binomtest


def accuracy(records: list[dict], key: str = "correct") -> float:
    if not records:
        return float("nan")
    return sum(bool(r[key]) for r in records) / len(records)


def flip_matrix(base: dict[str, bool], new: dict[str, bool]) -> dict:
    """base/new: uid -> correct. Only uids present in both are compared."""
    uids = sorted(set(base) & set(new))
    n11 = sum(base[u] and new[u] for u in uids)          # right -> right
    n00 = sum(not base[u] and not new[u] for u in uids)  # wrong -> wrong
    n10 = sum(base[u] and not new[u] for u in uids)      # right -> wrong (lost)
    n01 = sum(not base[u] and new[u] for u in uids)      # wrong -> right (gained)
    return {"n": len(uids), "both_right": n11, "both_wrong": n00,
            "lost": n10, "gained": n01}


def mcnemar_exact(n01: int, n10: int) -> float:
    """Exact McNemar p-value (two-sided binomial on the discordant pairs)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)


def compare_conditions(base_records: list[dict], new_records: list[dict]) -> dict:
    base = {r["uid"]: bool(r["correct"]) for r in base_records}
    new = {r["uid"]: bool(r["correct"]) for r in new_records}
    fm = flip_matrix(base, new)
    uids = set(base) & set(new)
    return {
        **fm,
        "acc_base": sum(base[u] for u in uids) / len(uids) if uids else float("nan"),
        "acc_new": sum(new[u] for u in uids) / len(uids) if uids else float("nan"),
        "p_mcnemar": mcnemar_exact(fm["gained"], fm["lost"]),
    }
