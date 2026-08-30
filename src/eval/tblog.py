from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from src.utils import REPO_ROOT


def tb_root(cfg: dict) -> Path:
    configured = os.environ.get(
        "CE_TF_LOG_DIR", cfg["paths"].get("tf_logs", "logs/tf"))
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def log_result(run_name: str, metrics: dict[str, float], cfg: dict,
               step: int = 0) -> None:
    root = tb_root(cfg)
    line = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {run_name} "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    with open(root / "results.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        from torch.utils.tensorboard import SummaryWriter

        w = SummaryWriter(log_dir=str(root / run_name))
        for k, v in metrics.items():
            w.add_scalar(k, v, global_step=step)
        w.close()
    except ImportError:
        pass  # text ledger already written
