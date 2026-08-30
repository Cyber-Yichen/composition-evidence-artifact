from __future__ import annotations

import numpy as np

FEATURE_DIM = 24


def featurize(f: dict) -> np.ndarray:
    s = f.get("subject") or {}
    e = s.get("edge") or {}
    b = f.get("balance") or {}
    sym = f.get("symmetry") or {}
    d = f.get("depth") or {}
    r = d.get("ratios") or {}
    ln = f.get("lines") or {}
    hr = f.get("headroom") or {}

    def g(dct, key, default=0.0):
        v = dct.get(key)
        return float(default if v is None else v)

    vec = np.array([
        g(s, "x", 0.5), g(s, "y", 0.5), g(s, "d_center"), g(s, "d_thirds"),
        g(s, "d_thirds_vlines"), g(s, "area_ratio"),
        g(e, "left"), g(e, "right"), g(e, "top"), g(e, "bottom"),
        g(b, "balance"), g(b, "mass_left", 0.5), g(b, "mass_right", 0.5),
        g(sym, "sym_h"), g(sym, "sym_v"),
        g(r, "foreground"), g(r, "mid"), g(r, "background"),
        g(ln, "n_leading") / 10.0,
        g(ln, "dominant_angle_deg") / 90.0,
        g(ln, "horizon_y"),
        g(hr, "face_top_y"),
        0.0 if f.get("no_subject") else 1.0,
        1.0 if f.get("depth") else 0.0,
    ], dtype=np.float32)
    assert vec.shape[0] == FEATURE_DIM
    return vec


def build_projector(hidden_size: int, k_tokens: int = 4, mlp_dim: int = 512):
    """2-layer MLP: FEATURE_DIM -> k_tokens * hidden_size. Torch import is lazy so
    the rest of the package works on CPU-only machines."""
    import torch.nn as nn

    class SoftTokenProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = k_tokens
            self.hidden = hidden_size
            self.net = nn.Sequential(
                nn.Linear(FEATURE_DIM, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, k_tokens * hidden_size),
            )

        def forward(self, feats):  # (B, FEATURE_DIM) -> (B, k, hidden)
            return self.net(feats).view(-1, self.k, self.hidden)

    return SoftTokenProjector()
