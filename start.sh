#!/bin/bash
set -e

# ============================================================
# DEPLOY: HunyuanOCR + LoRA | RunPod | vLLM OpenAI-Compatible
# Ref: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
# Use case: OCR
# ============================================================

# ===== CHỈNH SỬA PHẦN NÀY =====
BASE_MODEL="tencent/HunyuanOCR"
LORA_REPO="newai-vn/newai-ocr-1B"
HF_TOKEN="hf_YOUR_TOKEN_HERE"                # Hugging Face token
LORA_NAME="newai-ocr"                        # tên model khi gọi API, đặt tùy ý
LORA_LOCAL="/workspace/lora_adapter"
PORT=8000
MAX_MODEL_LEN=8192               
MAX_LORA_RANK=64                             # if your LoRA adapters have ranks [16, 32, 64], use --max-lora-rank 64 rather than 256
# ================================

echo "========================================================"
echo "  HunyuanOCR + LoRA Deploy — $(date)"
echo "  Base : $BASE_MODEL"
echo "  LoRA : $LORA_REPO  →  $LORA_NAME"
echo "========================================================"

# ── 1. Cài vLLM ──
echo ""
echo "[1/4] Installing vLLM..."
pip install uv -q
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
echo "✓ vLLM ready"

# ── 2. Cài transformers đúng commit HunyuanOCR yêu cầu ──
echo ""
echo "[2/4] Installing compatible transformers..."
pip install -q \
    "git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4" \
    huggingface_hub peft pillow
echo "✓ Dependencies ready"

# ── 3. Download LoRA adapter & tự động fix config ───
echo ""
echo "[3/4] Downloading LoRA adapter..."

python3 - <<PYEOF
import json, os
from huggingface_hub import snapshot_download

local_dir = "${LORA_LOCAL}"
snapshot_download(
    repo_id="${LORA_REPO}",
    local_dir=local_dir,
    token="${HF_TOKEN}",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)

# Fix base_model_name_or_path nếu trống (vLLM cần field này)
config_path = os.path.join(local_dir, "adapter_config.json")
with open(config_path) as f:
    cfg = json.load(f)

if not cfg.get("base_model_name_or_path"):
    cfg["base_model_name_or_path"] = "${BASE_MODEL}"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"✓ Fixed base_model_name_or_path → ${BASE_MODEL}")
else:
    print(f"✓ base_model_name_or_path: {cfg['base_model_name_or_path']}")

print(f"✓ LoRA adapter ready: {local_dir}")
PYEOF

# Gennerate API key
API_KEY="newwai_ocr_$(python3 -c "import secrets; print(secrets.token_hex(32))")"
echo "API_KEY: $API_KEY"

# ── 4. Jalankan vLLM server ──
echo ""
echo "[4/4] Starting vLLM OpenAI-compatible server..."
echo "  POST http://0.0.0.0:${PORT}/v1/chat/completions"
echo "  model = \"${LORA_NAME}\""
echo ""

# Dùng "vllm serve" theo đúng docs mới nhất
# Ref: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
#      https://docs.vllm.ai/projects/recipes/en/latest/Tencent-Hunyuan/HunyuanOCR.html
vllm serve "${BASE_MODEL}" \
    --port "${PORT}" \
    --enable-lora \
    --lora-modules "${LORA_NAME}=${LORA_LOCAL}" \
    --max-lora-rank "${MAX_LORA_RANK}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt "image=1" \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --hf-token "${HF_TOKEN}"\
    --api-key "${API_KEY}"
