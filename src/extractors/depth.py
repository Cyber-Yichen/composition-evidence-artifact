from __future__ import annotations

import numpy as np

from src.extractors.base import CachedExtractor


class DepthExtractor(CachedExtractor):
    name = "depth"
    storage = "npz"

    def __init__(self, cache_dir, model_id: str, larger_is_closer: bool = True,
                 device: str | None = None, store_size: int = 384):
        super().__init__(cache_dir)
        self.model_id = model_id
        self.larger_is_closer = larger_is_closer
        self.device = device
        self.store_size = store_size  # cache at reduced resolution; enough for terciles
        self._pipe = None

    def _ensure_model(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = pipeline("depth-estimation", model=self.model_id,
                              device=0 if self.device == "cuda" else -1)

    def extract(self, image_bgr: np.ndarray) -> dict:
        import cv2
        from PIL import Image

        self._ensure_model()
        pred = self._pipe(Image.fromarray(image_bgr[:, :, ::-1]))["predicted_depth"]
        d = pred.squeeze().float().cpu().numpy()
        if not self.larger_is_closer:
            d = -d
        d = (d - d.min()) / (d.max() - d.min() + 1e-9)
        h, w = d.shape
        if max(h, w) > self.store_size:
            s = self.store_size / max(h, w)
            d = cv2.resize(d, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return {"closeness": d.astype(np.float16)}
