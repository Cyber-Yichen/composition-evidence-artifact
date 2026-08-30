from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class CachedExtractor(ABC):
    name: str = "base"
    storage: str = "json"  # "json" | "npz"

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir) / self.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, image_bgr: np.ndarray, key: str):
        f = self.cache_dir / f"{key}.{ 'npz' if self.storage == 'npz' else 'json'}"
        if f.exists():
            return self._load(f)
        out = self.extract(image_bgr)
        self._save(f, out)
        return out

    @abstractmethod
    def extract(self, image_bgr: np.ndarray):
        ...

    def _save(self, f: Path, out) -> None:
        if self.storage == "npz":
            np.savez_compressed(f, **out)
        else:
            f.write_text(json.dumps(out), encoding="utf-8")

    def _load(self, f: Path):
        if self.storage == "npz":
            with np.load(f) as z:
                return {k: z[k] for k in z.files}
        return json.loads(f.read_text(encoding="utf-8"))
