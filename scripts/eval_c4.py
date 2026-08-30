from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from train_projector import PLACEHOLDER  # noqa: E402  (same placeholder token)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/projector/best.pt")
    ap.add_argument("--k-tokens", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="C4", help="condition tag for the output file")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from qwen_vl_utils import process_vision_info
    from rich.progress import track
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.eval.datasets import load_shotbench, sc_subset
    from src.eval.parsing import parse_choice
    from src.eval.tblog import log_result
    from src.injection.soft_token import build_projector, featurize
    from src.utils import (append_jsonl, done_uids, load_config, load_yaml,
                           read_jsonl, set_seed)

    cfg = load_config()
    set_seed(cfg["seed"])
    models = load_yaml(REPO / "configs" / "models.yaml")
    hf_id = models["local"]["qwen2_5_vl_7b"]["hf_id"]
    root = REPO / cfg["paths"]["shotbench"]

    ev = read_jsonl(REPO / cfg["paths"]["results"] / "evidence" / "shotbench_sc.jsonl")
    feats = {r["uid"]: r["features"] for r in ev}
    samples = [s for s in sc_subset(load_shotbench(root)) if s.uid in feats]
    if args.limit:
        samples = samples[: args.limit]

    processor = AutoProcessor.from_pretrained(hf_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa").eval()
    hidden = model.config.hidden_size
    projector = build_projector(hidden, args.k_tokens).to(args.device)
    projector.load_state_dict(torch.load(REPO / args.ckpt, map_location=args.device))
    projector.eval()
    ph_id = processor.tokenizer.convert_tokens_to_ids(PLACEHOLDER)

    state = {"soft": None}
    hidden_size = hidden

    def hook(module, inputs, output):
        if state["soft"] is None:
            return output
        mask = inputs[0] == ph_id
        if mask.any():
            output = output.clone()
            output[mask] = state["soft"].reshape(-1, hidden_size)[: mask.sum()]
        return output

    model.get_input_embeddings().register_forward_hook(hook)

    out_path = (REPO / cfg["paths"]["results"] / "eval" / f"qwen2_5_vl_7b_{args.tag}_sc.jsonl")
    done = done_uids(out_path)
    for s in track(samples, description="C4"):
        if s.uid in done:
            continue
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(s.options.items()))
        user = (PLACEHOLDER * args.k_tokens
                + f"Question: {s.question}\nOptions:\n{opts}\n"
                  "Please select the most likely answer from the options above.")
        img = str((root / s.media_paths[0]).resolve())
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{img}"},
            {"type": "text", "text": user}]}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, _ = process_vision_info(msgs)
        enc = processor(text=[text], images=image_inputs,
                        return_tensors="pt").to("cuda:0")
        vec = torch.tensor([featurize(feats[s.uid])], dtype=torch.float32,
                           device=args.device)
        with torch.no_grad():
            state["soft"] = projector(vec).to(torch.bfloat16)
            out = model.generate(**enc, max_new_tokens=cfg["eval"]["max_new_tokens"],
                                 do_sample=False)
            state["soft"] = None
        resp = processor.batch_decode(out[:, enc.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0]
        letter, method = parse_choice(resp, s.options)
        append_jsonl(out_path, {"uid": s.uid, "model": "qwen2_5_vl_7b",
                                "condition": args.tag, "category": s.category,
                                "response": resp, "parsed": letter,
                                "parse_method": method, "answer": s.answer,
                                "correct": letter == s.answer})
    rows = read_jsonl(out_path)
    acc = sum(r["correct"] for r in rows) / len(rows)
    print(f"{args.tag}: acc={acc:.4f} (n={len(rows)})")
    log_result(f"qwen2_5_vl_7b_{args.tag}_sc", {"accuracy": acc}, cfg)


if __name__ == "__main__":
    main()
