#!/bin/bash
set -e

# ============================================================
# DEPLOY: HunyuanOCR + LoRA | RunPod | vLLM OpenAI-Compatible
# Ref: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
# Use case: OCR sổ đỏ
# ============================================================

# ===== CHỈNH SỬA PHẦN NÀY =====
BASE_MODEL="tencent/HunyuanOCR"
LORA_REPO="your-hf-username/your-lora-repo"  # HuggingFace repo LoRA của bạn
HF_TOKEN="hf_YOUR_TOKEN_HERE"
LORA_NAME="so-do-ocr"          # tên model khi gọi API, đặt tùy ý
LORA_LOCAL="/workspace/lora_adapter"
PORT=8000
GPU_UTIL=0.85
MAX_MODEL_LEN=8192              # nới rộng cho sổ đỏ (tọa độ + text dài)
MAX_LORA_RANK=16                # khớp với r=16 trong adapter_config.json
# ================================

echo "========================================================"
echo "  HunyuanOCR + LoRA Deploy — $(date)"
echo "  Base : $BASE_MODEL"
echo "  LoRA : $LORA_REPO  →  $LORA_NAME"
echo "========================================================"

# ── 1. Cài vLLM nightly (bắt buộc cho HunyuanOCR) ──────────
echo ""
echo "[1/4] Installing vLLM nightly..."
pip install uv -q
uv pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly -q
echo "✓ vLLM ready"

# ── 2. Cài transformers đúng commit HunyuanOCR yêu cầu ──────
echo ""
echo "[2/4] Installing compatible transformers..."
pip install -q \
    "git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4" \
    huggingface_hub peft pillow
echo "✓ Dependencies ready"

# ── 3. Download LoRA adapter & tự động fix config ───────────
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

# ── 4. Jalankan vLLM server (vllm serve — cú pháp mới nhất) ─
echo ""
echo "[4/4] Starting vLLM OpenAI-compatible server..."
echo "  POST http://0.0.0.0:${PORT}/v1/chat/completions"
echo "  model = \"${LORA_NAME}\""
echo ""

# Dùng "vllm serve" theo đúng docs mới nhất
# Ref: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
vllm serve "${BASE_MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --enable-lora \
    --lora-modules "${LORA_NAME}=${LORA_LOCAL}" \
    --max-lora-rank "${MAX_LORA_RANK}" \
    --max-loras 1 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt "image=1" \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --dtype bfloat16 \
    --trust-remote-code \
    --hf-token "${HF_TOKEN}"
