from __future__ import annotations

import numpy as np


def frame_indices(n_total: int, n_uniform: int = 4) -> tuple[list[int], int]:
    if n_total <= 0:
        return [], -1
    uniform = np.linspace(0, n_total - 1, n_uniform).round().astype(int).tolist()
    middle = (n_total - 1) // 2
    idxs = sorted(set(uniform) | {middle})
    return idxs, middle


def sample_frames(video_path: str, n_uniform: int = 4):
    """Returns (frames_bgr: list[np.ndarray], indices: list[int], middle_pos: int)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs, middle = frame_indices(n_total, n_uniform)
    frames, kept = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
            kept.append(i)
    cap.release()
    middle_pos = kept.index(middle) if middle in kept else len(kept) // 2
    return frames, kept, middle_pos
