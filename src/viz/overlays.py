from __future__ import annotations

import numpy as np


def draw_overlay(image_bgr: np.ndarray, boxes: list[dict] | None = None,
                 segments: list[dict] | None = None,
                 subject_box: list[float] | None = None) -> np.ndarray:
    import cv2

    img = image_bgr.copy()
    h, w = img.shape[:2]

    for fx in (1 / 3, 2 / 3):  # thirds grid
        cv2.line(img, (int(fx * w), 0), (int(fx * w), h), (255, 255, 255), 1)
        cv2.line(img, (0, int(fx * h)), (w, int(fx * h)), (255, 255, 255), 1)

    for s in (segments or [])[:10]:
        p1 = (int(s["p1"][0] * w), int(s["p1"][1] * h))
        p2 = (int(s["p2"][0] * w), int(s["p2"][1] * h))
        cv2.line(img, p1, p2, (0, 255, 255), 2)

    for b in boxes or []:
        x1, y1, x2, y2 = b["box"]
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                      (0, 200, 0), 2)

    if subject_box is not None:
        x1, y1, x2, y2 = subject_box
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                      (0, 0, 255), 3)
    return img


def depth_pseudocolor(closeness: np.ndarray) -> np.ndarray:
    import cv2

    d = (np.asarray(closeness, dtype=np.float32) * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


def triptych(image_bgr: np.ndarray, overlay_bgr: np.ndarray,
             depth_bgr: np.ndarray | None, out_path: str, titles=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [image_bgr, overlay_bgr] + ([depth_bgr] if depth_bgr is not None else [])
    titles = titles or ["original", "evidence overlay", "depth"][:len(panels)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel[:, :, ::-1])
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
