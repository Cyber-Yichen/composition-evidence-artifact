from __future__ import annotations

import numpy as np


def confusion_matrix(records: list[dict], labels: list[str],
                     option_texts: dict[str, dict[str, str]]) -> np.ndarray:
    """records: eval jsonl rows; option_texts: uid -> options dict.
    Rows = true label text, cols = predicted label text (unparsed -> ignored)."""
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for r in records:
        opts = option_texts.get(r["uid"], {})
        true_text = opts.get(r["answer"])
        pred_text = opts.get(r["parsed"]) if r["parsed"] else None
        if true_text in idx and pred_text in idx:
            m[idx[true_text], idx[pred_text]] += 1
    return m


def plot_confusion(m: np.ndarray, labels: list[str], title: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def flip_buckets(base_records: list[dict], new_records: list[dict],
                 d_thirds_by_uid: dict[str, float],
                 bins=(0.0, 0.05, 0.10, 0.15, 0.25, 1.0)) -> list[dict]:
    """Bucket samples by geometric clarity and measure where gains concentrate."""
    base = {r["uid"]: bool(r["correct"]) for r in base_records}
    new = {r["uid"]: bool(r["correct"]) for r in new_records}
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        uids = [u for u, d in d_thirds_by_uid.items()
                if lo <= d < hi and u in base and u in new]
        if not uids:
            continue
        rows.append({
            "bucket": f"[{lo:.2f},{hi:.2f})",
            "n": len(uids),
            "acc_base": sum(base[u] for u in uids) / len(uids),
            "acc_new": sum(new[u] for u in uids) / len(uids),
        })
    return rows
