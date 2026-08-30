from __future__ import annotations

from pathlib import Path

from rich.progress import track

from src.eval.parsing import parse_choice
from src.extractors.frames import sample_frames
from src.injection.prompts import build_prompt
from src.utils import append_jsonl, done_uids


def media_for_sample(sample, bench_root: str | Path, n_uniform: int = 4,
                     frame_dir: str | Path | None = None) -> list[Path]:
    """Images pass through; videos are sampled to frames cached under frame_dir."""
    paths = sample.resolve_media(bench_root)
    if sample.media_type == "image":
        return paths
    import cv2

    frame_dir = Path(frame_dir or Path(bench_root) / "frames") / sample.uid
    frame_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frame_dir.glob("*.jpg"))
    if existing:
        return existing
    frames, kept, _ = sample_frames(str(paths[0]), n_uniform)
    out = []
    for idx, frame in zip(kept, frames):
        p = frame_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(p), frame)
        out.append(p)
    return out


def run_condition(samples, backend, condition: str, out_path: str | Path,
                  model_name: str, cfg: dict, prompts_cfg: dict,
                  evidence_texts: dict[str, str] | None = None,
                  bench_root: str | Path = "data/shotbench",
                  position: str | None = None) -> None:
    """evidence_texts: uid -> rendered evidence block (required for C2/C3)."""
    done = done_uids(out_path)
    position = position or cfg["evidence"]["position"]
    for s in track(samples, description=f"{model_name} {condition}"):
        if s.uid in done:
            continue
        ev = (evidence_texts or {}).get(s.uid)
        prompt = build_prompt(s.question, s.options, condition, prompts_cfg,
                              evidence_text=ev, position=position)
        media = media_for_sample(s, bench_root, cfg["video"]["n_frames"] - 1)
        response = backend.generate(media, prompt)
        letter, method = parse_choice(response, s.options)
        append_jsonl(out_path, {
            "uid": s.uid,
            "model": model_name,
            "condition": condition,
            "category": s.category,
            "response": response,
            "parsed": letter,
            "parse_method": method,
            "answer": s.answer,
            "correct": letter == s.answer,
        })
