from pathlib import Path

import yaml

from src.eval.backends import _official_alibaba_base_url

REPO = Path(__file__).resolve().parents[1]


def test_commercial_models_use_explicit_official_providers():
    models = yaml.safe_load((REPO / "configs" / "models.yaml").read_text())
    expected = {
        "qwen_vl_max": "alibaba_model_studio",
        "qwen3_vl_plus": "alibaba_model_studio",
        "gpt_5_4": "openai",
        "gemini_3_1_pro": "google",
        "claude_opus_4_8": "anthropic",
    }
    assert {name: cfg["provider"] for name, cfg in models["api"].items()} == expected
    assert all(cfg["evaluation_month"] == "2026-07"
               for cfg in models["api"].values())


def test_alibaba_backend_rejects_third_party_hosts():
    assert _official_alibaba_base_url(
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    assert _official_alibaba_base_url(
        "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
    assert not _official_alibaba_base_url("http://dashscope.aliyuncs.com/v1")
    assert not _official_alibaba_base_url("https://example.com/v1")


def test_legacy_generic_endpoint_variables_are_absent():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO / "configs" / "models.yaml",
            REPO / "scripts" / "env.sh",
            REPO / "src" / "eval" / "backends.py",
        )
    )
    legacy_primary = "CE" + "_API_BASE_URL"
    legacy_secondary = "CE2" + "_API_BASE_URL"
    assert legacy_primary not in text
    assert legacy_secondary not in text
    assert "OpenAICompatBackend" not in text
