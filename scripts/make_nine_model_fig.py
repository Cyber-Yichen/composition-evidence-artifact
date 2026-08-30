from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    ("Qwen2.5-VL-7B", 49.5, 56.2),
    ("InternVL3-8B", 45.5, 47.8),
    ("ShotVL-7B v1.1", 62.2, 63.7),
    ("gpt-5.4", 59.1, 61.2),
    ("gemini-3.1-pro", 59.3, 60.5),
    ("qwen3-vl-plus", 49.7, 49.9),
    ("qwen-vl-max", 53.7, 52.8),
    ("LLaVA-OV-7B", 43.8, 39.5),
    ("claude-opus-4-8", 63.0, 56.2),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    x = np.arange(len(MODELS))
    c0 = [m[1] for m in MODELS]
    fin = [m[2] for m in MODELS]
    b1 = ax.bar(x - 0.2, c0, 0.38, label="C0 zero-shot", color="#93c5fd",
                edgecolor="#1e3a5f", linewidth=0.6)
    b2 = ax.bar(x + 0.2, fin, 0.38, label="+ evidence (final recipe)",
                color="#fdba74", edgecolor="#7c2d12", linewidth=0.6)
    ax.bar_label(b1, fmt="%.1f", fontsize=7.5, padding=1.5)
    ax.bar_label(b2, fmt="%.1f", fontsize=7.5, padding=1.5)
    ax.set_xticks(x, [m[0] for m in MODELS], rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("ShotBench SC accuracy (%)")
    ax.set_ylim(0, 74)  # headroom so labels and legend never touch the bars
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("figures/nine_model_direction.pdf", bbox_inches="tight")
    fig.savefig("figures/nine_model_direction.png", dpi=200, bbox_inches="tight")
    print("saved figures/nine_model_direction.{pdf,png}")


if __name__ == "__main__":
    main()
