from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Hard-pin every cache inside the project even if env.sh wasn't sourced.
os.environ.setdefault("HF_HOME", str(REPO / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO / ".cache"))

SHOTBENCH_REPO = "Vchitect/ShotBench"
SHOTQA_REPO = "Vchitect/ShotQA"
# RefineShot has no separate media; it restructures ShotBench
# options at eval time. Code: https://github.com/wuhang03/RefineShot
# (category.json + extract.py). Integration = port their restructuring into
# src/eval/datasets.py, reusing the already-downloaded ShotBench media.
REFINESHOT_CANDIDATES: list[str] = []  # intentionally empty — github code, not a dataset
# CineTechBench annotations are on HF; media must be fetched from
# per-image links in image_annotation (copyright). https://github.com/PRIS-CV/CineTechBench
CINETECHBENCH_CANDIDATES: list[str] = ["Xinran0906/CineTechBench"]


def download_snapshot(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      local_dir=str(local_dir))


def extract_tars(local_dir: Path) -> None:
    for tar_path in sorted(local_dir.glob("*.tar")):
        marker = local_dir / f".extracted_{tar_path.stem}"
        if marker.exists():
            continue
        print(f"extracting {tar_path.name} ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(local_dir, filter="data")
        marker.touch()


def get_shotbench() -> None:
    d = REPO / "data" / "shotbench"
    download_snapshot(SHOTBENCH_REPO, d)
    extract_tars(d)
    print("shotbench ready:", sorted(p.name for p in d.iterdir()))


def get_shotqa() -> None:
    d = REPO / "data" / "shotqa"
    download_snapshot(SHOTQA_REPO, d)
    print("shotqa ready")


def get_pending(name: str, candidates: list[str]) -> None:
    if not candidates:
        print(f"{name}: no standalone dataset download is defined; see data/README.md.")
        return
    d = REPO / "data" / name
    for c in candidates:
        try:
            download_snapshot(c, d)
            print(f"{name} ready from {c}")
            return
        except Exception as e:  # noqa: BLE001 — try next candidate
            print(f"{name}: candidate {c} failed ({type(e).__name__})")
    print(f"{name}: all candidates failed; check network and repository availability.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="shotbench",
                    choices=["shotbench", "shotqa", "refineshot",
                             "cinetechbench", "all"])
    args = ap.parse_args()
    if args.which in ("shotbench", "all"):
        get_shotbench()
    if args.which in ("shotqa", "all"):
        get_shotqa()
    if args.which in ("refineshot", "all"):
        get_pending("refineshot", REFINESHOT_CANDIDATES)
    if args.which in ("cinetechbench", "all"):
        get_pending("cinetechbench", CINETECHBENCH_CANDIDATES)


if __name__ == "__main__":
    main()
