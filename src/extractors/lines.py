from __future__ import annotations

import math

import cv2
import numpy as np

from src.extractors.base import CachedExtractor


def _to_segments(raw: np.ndarray, w: int, h: int) -> list[dict]:
    segs = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4).astype(float):
        p1 = [x1 / w, y1 / h]
        p2 = [x2 / w, y2 / h]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        angle = angle - 180 if angle >= 90 else (angle + 180 if angle < -90 else angle)
        segs.append({"p1": p1, "p2": p2, "length": length, "angle_deg": angle})
    return segs


def detect_segments(gray: np.ndarray, detector: str = "lsd") -> list[dict]:
    h, w = gray.shape
    if detector == "lsd":
        try:
            lsd = cv2.createLineSegmentDetector()
            raw = lsd.detect(gray)[0]
            if raw is None:
                return []
            return _to_segments(raw, w, h)
        except cv2.error:
            pass  # LSD unavailable in this OpenCV build -> Hough fallback
    edges = cv2.Canny(gray, 50, 150)
    min_len = int(0.08 * min(w, h))
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                          minLineLength=min_len, maxLineGap=10)
    return [] if raw is None else _to_segments(raw, w, h)


class LineExtractor(CachedExtractor):
    name = "lines"
    storage = "json"

    def __init__(self, cache_dir, detector: str = "lsd"):
        super().__init__(cache_dir)
        self.detector = detector

    def extract(self, image_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return {"segments": detect_segments(gray, self.detector)}
