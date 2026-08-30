from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse


# official ShotBench reasoning-mode system prompt (evaluate.py, used by ShotVL)
REASONING_SYSTEM = (
    "A conversation between User and Assistant. The user asks a question, and "
    "the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The "
    "reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags.")


def build_backend(name: str, mcfg: dict, gen_cfg: dict):
    if "hf_id" in mcfg:
        cls = QwenVLBackend if mcfg.get("backend") == "qwen_vl" else GenericHFBackend
        reasoning = mcfg.get("reasoning", False)
        return cls(mcfg["hf_id"], dtype=mcfg.get("dtype", "bfloat16"),
                   trust_remote_code=mcfg.get("trust_remote_code", False),
                   max_new_tokens=1024 if reasoning else gen_cfg["max_new_tokens"],
                   system_prompt=REASONING_SYSTEM if reasoning else None)
    provider = mcfg.get("provider")
    kwargs = {
        "model": mcfg["model"],
        "api_key_env": mcfg["api_key_env"],
        "max_tokens": mcfg.get("max_tokens", gen_cfg["max_new_tokens"]),
    }
    if provider == "openai":
        return OpenAIOfficialBackend(**kwargs)
    if provider == "google":
        return GeminiOfficialBackend(**kwargs)
    if provider == "anthropic":
        return AnthropicOfficialBackend(**kwargs)
    if provider == "alibaba_model_studio":
        return AlibabaModelStudioBackend(
            **kwargs, base_url_env=mcfg["base_url_env"])
    raise ValueError(
        f"{name}: unsupported or missing official API provider {provider!r}")


class QwenVLBackend:
    """Qwen2.5-VL family (incl. ShotVL, which is Qwen2.5-VL based)."""

    def __init__(self, hf_id: str, dtype: str = "bfloat16",
                 trust_remote_code: bool = False, max_new_tokens: int = 64,
                 system_prompt: str | None = None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.processor = AutoProcessor.from_pretrained(hf_id)
        try:
            import flash_attn  # noqa: F401

            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=getattr(torch, dtype), device_map="auto",
            attn_implementation=attn).eval()

    def generate(self, image_paths: list[str | Path], prompt: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        content = [{"type": "image", "image": f"file://{p}"} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


class GenericHFBackend:
    """Fallback for InternVL3 / LLaVA-OneVision via AutoModelForImageTextToText +
    chat template. Verify each model with a small smoke test before evaluation."""

    def __init__(self, hf_id: str, dtype: str = "bfloat16",
                 trust_remote_code: bool = False, max_new_tokens: int = 64,
                 system_prompt: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.processor = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_id, torch_dtype=getattr(torch, dtype), device_map="auto",
            trust_remote_code=trust_remote_code).eval()

    def generate(self, image_paths: list[str | Path], prompt: str) -> str:
        import torch
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths]
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.processor(text=text, images=images,
                                return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"API credential missing: export {name} in the shell "
            "(never write credentials into files).")
    return value


def _image_media_type(path: str | Path) -> str:
    media_type = mimetypes.guess_type(str(path))[0]
    supported = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    if media_type not in supported:
        raise ValueError(
            f"unsupported image type for official APIs: {path} ({media_type})")
    return media_type


def _image_b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _image_data_url(path: str | Path) -> str:
    return f"data:{_image_media_type(path)};base64,{_image_b64(path)}"


def _official_alibaba_base_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        host == "aliyuncs.com" or host.endswith(".aliyuncs.com"))


class OpenAIOfficialBackend:
    """OpenAI Responses API through the official OpenAI Python SDK."""

    def __init__(self, model: str, api_key_env: str,
                 max_tokens: int = 64, max_retries: int = 6):
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=_require_env(api_key_env), max_retries=max_retries)

    def generate(self, image_paths: list[str | Path], prompt: str) -> str:
        content = [
            {"type": "input_image", "image_url": _image_data_url(path)}
            for path in image_paths
        ]
        content.append({"type": "input_text", "text": prompt})
        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            temperature=0.0,
            max_output_tokens=self.max_tokens,
        )
        return response.output_text or ""


class GeminiOfficialBackend:
    """Gemini generateContent API through Google's official google-genai SDK."""

    def __init__(self, model: str, api_key_env: str,
                 max_tokens: int = 64):
        from google import genai
        from google.genai import types

        self.model = model
        self.max_tokens = max_tokens
        self.types = types
        self.client = genai.Client(api_key=_require_env(api_key_env))

    def generate(self, image_paths: list[str | Path], prompt: str) -> str:
        contents = [
            self.types.Part.from_bytes(
                data=Path(path).read_bytes(),
                mime_type=_image_media_type(path),
            )
            for path in image_paths
        ]
        contents.append(prompt)
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self.types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text or ""


class AnthropicOfficialBackend:
    """Claude Messages API through Anthropic's official Python SDK."""

    def __init__(self, model: str, api_key_env: str,
                 max_tokens: int = 64, max_retries: int = 6):
        from anthropic import Anthropic

        self.model = model
        self.max_tokens = max_tokens
        self.client = Anthropic(
            api_key=_require_env(api_key_env), max_retries=max_retries)

    def generate(self, image_paths: list[str | Path], prompt: str) -> str:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _image_media_type(path),
                    "data": _image_b64(path),
                },
            }
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text")


class AlibabaModelStudioBackend:
    """Official Alibaba Model Studio OpenAI-compatible multimodal API."""

    def __init__(self, model: str, base_url_env: str, api_key_env: str,
                 max_tokens: int = 64, max_retries: int = 6):
        from openai import OpenAI

        self.model = model
        self.max_tokens = max_tokens
        base_url = _require_env(base_url_env).rstrip("/")
        if not _official_alibaba_base_url(base_url):
            raise RuntimeError(
                f"{base_url_env} must be an official HTTPS aliyuncs.com "
                "Model Studio endpoint.")
        self.client = OpenAI(
            api_key=_require_env(api_key_env),
            base_url=base_url,
            max_retries=max_retries,
        )

    def generate(self, image_paths: list[str | Path], prompt: str) -> str:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(path)},
            }
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""
