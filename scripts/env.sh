# Source this before Python, package-manager, or model-download commands:
#   source scripts/env.sh
# This keeps tool and model caches inside the project directory.

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$PROJECT_ROOT/.cache/huggingface/hub"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.cache/uv-python"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"

# Use the classic Hugging Face HTTP download path for broad proxy compatibility.
export HF_HUB_DISABLE_XET=1

# Optional Hugging Face mirror:
# export HF_ENDPOINT=https://hf-mirror.com

# Optional credentials for official commercial-model APIs. Never commit real keys.
#   export OPENAI_API_KEY="<api-key>"
#   export GEMINI_API_KEY="<api-key>"
#   export ANTHROPIC_API_KEY="<api-key>"
#   export DASHSCOPE_API_KEY="<api-key>"
# Alibaba Model Studio URLs vary by account region/workspace. The backend rejects
# non-HTTPS hosts outside the official aliyuncs.com domain.
#   export DASHSCOPE_BASE_URL="https://<workspace>.<region>.maas.aliyuncs.com/compatible-mode/v1"
