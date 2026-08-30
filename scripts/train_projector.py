from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PLACEHOLDER = "<|box_start|>"  # existing rare special token, embeddings replaced by hook


def build_batch_texts(processor, items, k_tokens):
    """Chat-formatted texts with k placeholder tokens before the question."""
    texts, image_paths = [], []
    for it in items:
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(it["options"].items()))
        user = (PLACEHOLDER * k_tokens
                + f"Question: What's the composition of this shot?\nOptions:\n{opts}\n"
                  "Please select the most likely answer from the options above.")
        msgs = [{"role": "user", "content": [
                    {"type": "image", "image": f"file://{it['abs_image']}"},
                    {"type": "text", "text": user}]},
                {"role": "assistant", "content": it["answer"]}]
        texts.append(processor.apply_chat_template(msgs, tokenize=False))
        image_paths.append(it["abs_image"])
    return texts, image_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=768 * 28 * 28,
                    help="vision token budget per training image")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k-tokens", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out-dir", default="results/projector",
                    help="checkpoint dir (use a distinct dir per k/config variant)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from qwen_vl_utils import process_vision_info
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from src.eval.tblog import tb_root
    from src.injection.soft_token import build_projector
    from src.utils import load_config, load_yaml, read_jsonl, set_seed

    cfg = load_config()
    set_seed(cfg["seed"])
    models = load_yaml(REPO / "configs" / "models.yaml")
    hf_id = models["local"]["qwen2_5_vl_7b"]["hf_id"]

    rows = read_jsonl(REPO / cfg["paths"]["results"] / "shotqa_features.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    root = REPO / cfg["paths"]["shotqa"]
    for r in rows:
        p = root / r["image"]
        r["abs_image"] = str(p if p.exists() else root / Path(r["image"]).name)
    rng = random.Random(cfg["seed"])
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"train {len(train_rows)} / val {n_val}")

    processor = AutoProcessor.from_pretrained(hf_id, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=args.device,
        attn_implementation="sdpa")
    model.requires_grad_(False)
    model.enable_input_require_grads()  # grads must flow into inputs_embeds
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()

    hidden = model.config.hidden_size
    # projector stays fp32 (AdamW on bf16 weights loses precision); outputs cast
    projector = build_projector(hidden, args.k_tokens).to(args.device)
    ph_id = processor.tokenizer.convert_tokens_to_ids(PLACEHOLDER)
    assert isinstance(ph_id, int) and ph_id >= 0, "placeholder token missing"

    # embedding hook: replace placeholder positions with projector outputs
    state = {"soft": None}
    emb_layer = model.get_input_embeddings()

    def hook(module, inputs, output):
        if state["soft"] is None:
            return output
        ids = inputs[0]
        mask = ids == ph_id
        if mask.any():
            output = output.clone()
            output[mask] = state["soft"].reshape(-1, hidden)[: mask.sum()]
        return output

    emb_layer.register_forward_hook(hook)

    opt = torch.optim.AdamW(projector.parameters(), lr=args.lr)
    steps_per_epoch = (len(train_rows) + args.bs - 1) // args.bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch)
    writer = SummaryWriter(log_dir=str(tb_root(cfg) / "projector_train"))
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def forward_batch(batch, train: bool):
        texts, _ = build_batch_texts(processor, batch, args.k_tokens)
        msgs_for_vision = [[{"role": "user", "content": [
            {"type": "image", "image": f"file://{it['abs_image']}"}]}]
            for it in batch]
        image_inputs = [process_vision_info(m)[0] for m in msgs_for_vision]
        image_inputs = [i for sub in image_inputs for i in (sub or [])]
        enc = processor(text=texts, images=image_inputs, return_tensors="pt",
                        padding=True).to("cuda:0")
        # labels: answer token(s) only — mask everything before the last
        # assistant turn (single letter + <|im_end|>)
        labels = enc.input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        for b in range(labels.shape[0]):
            ids = enc.input_ids[b].tolist()
            # find last assistant letter: last non-pad, non-special run of len 1
            # simpler: mask all but the final 3 non-pad tokens (letter, im_end, \n)
            nonpad = [i for i, t in enumerate(ids)
                      if t != processor.tokenizer.pad_token_id]
            keep = set(nonpad[-3:])
            for i in range(labels.shape[1]):
                if i not in keep:
                    labels[b, i] = -100
        vecs = torch.tensor([it["vec"] for it in batch], dtype=torch.float32,
                            device=args.device)
        state["soft"] = projector(vecs).to(torch.bfloat16)
        out = model(**enc, labels=labels)
        state["soft"] = None
        return out.loss

    step = 0
    best_val = float("inf")
    for epoch in range(args.epochs):
        rng.shuffle(train_rows)
        for i in range(0, len(train_rows), args.bs):
            batch = train_rows[i:i + args.bs]
            loss = forward_batch(batch, train=True) / args.grad_accum
            loss.backward()
            if (i // args.bs + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            loss = loss * args.grad_accum  # undo scaling for logging
            step += 1
            if step % 20 == 0:
                writer.add_scalar("loss/train", loss.item(), step)
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}",
                      flush=True)
            if step % 200 == 0:
                with torch.no_grad():
                    vl = sum(forward_batch(val_rows[j:j + args.bs], False).item()
                             for j in range(0, len(val_rows), args.bs))
                vl /= max(1, (len(val_rows) + args.bs - 1) // args.bs)
                writer.add_scalar("loss/val", vl, step)
                print(f"  val loss {vl:.4f}", flush=True)
                if vl < best_val:
                    best_val = vl
                    torch.save(projector.state_dict(), out_dir / "best.pt")
        torch.save(projector.state_dict(), out_dir / f"epoch{epoch}.pt")
    torch.save(projector.state_dict(), out_dir / "projector.pt")
    writer.close()
    print(f"done. best val loss {best_val:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
