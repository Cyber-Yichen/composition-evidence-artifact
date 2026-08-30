from __future__ import annotations

import numpy as np

from src.extractors.base import CachedExtractor


class GroundingDINOExtractor(CachedExtractor):
    name = "grounding_dino"
    storage = "json"

    def __init__(self, cache_dir, model_id: str, prompts: list[str],
                 box_threshold: float, text_threshold: float, device: str | None = None):
        super().__init__(cache_dir)
        self.model_id = model_id
        self.prompts = prompts
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id).to(self.device).eval()

    def extract(self, image_bgr: np.ndarray) -> dict:
        import torch
        from PIL import Image

        self._ensure_model()
        image = Image.fromarray(image_bgr[:, :, ::-1])  # BGR -> RGB
        h, w = image_bgr.shape[:2]
        boxes = []
        for round_idx, text in enumerate(self.prompts):
            inputs = self._processor(images=image, text=text,
                                     return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            res = self._processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[(h, w)])[0]
            for box, score, phrase in zip(res["boxes"], res["scores"],
                                          res.get("text_labels", res.get("labels"))):
                x1, y1, x2, y2 = [float(v) for v in box]
                boxes.append({
                    "box": [x1 / w, y1 / h, x2 / w, y2 / h],
                    "score": float(score),
                    "phrase": str(phrase),
                    "prompt_round": round_idx,
                })
        return {"boxes": boxes}
