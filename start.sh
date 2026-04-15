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
HF_TOKEN="${HF_TOKEN}"
LORA_NAME="newai-ocr"
LORA_LOCAL="/workspace/lora_adapter"
PORT=8000
MAX_MODEL_LEN=8192
MAX_LORA_RANK=64
# ================================

echo "========================================================"
echo "  HunyuanOCR + LoRA Deploy — $(date)"
echo "  Base : $BASE_MODEL"
echo "  LoRA : $LORA_REPO  →  $LORA_NAME"
echo "========================================================"

# ── 1. Download LoRA adapter (bỏ qua nếu đã có) ─────────────
echo ""
if [ ! -d "$LORA_LOCAL" ]; then
    echo "[1/2] Downloading LoRA adapter..."

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

# Fix base_model_name_or_path nếu trống
config_path = os.path.join(local_dir, "adapter_config.json")
with open(config_path) as f:
    cfg = json.load(f)

if not cfg.get("base_model_name_or_path"):
    cfg["base_model_name_or_path"] = "${BASE_MODEL}"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("✓ Fixed base_model_name_or_path → ${BASE_MODEL}")
else:
    print(f"✓ base_model_name_or_path: {cfg['base_model_name_or_path']}")

print(f"✓ LoRA adapter ready: {local_dir}")
PYEOF
else
    echo "[1/2] LoRA adapter already exists, skipping download..."
fi

# ── 2. Start vLLM server ─────────────────────────────────────
echo ""
echo "[2/2] Starting vLLM OpenAI-compatible server..."
echo "  model = \"${LORA_NAME}\""
echo ""

# Generate API key
API_KEY="newai_ocr_$(python3 -c "import secrets; print(secrets.token_hex(32))")"
echo "================================================================"
echo "  API_KEY: $API_KEY"
echo "  (copy lại key này để dùng trong client)"
echo "================================================================"
echo ""

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
    --hf-token "${HF_TOKEN}" \
    --api-key "${API_KEY}"
