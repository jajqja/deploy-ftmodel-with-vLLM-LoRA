# ============================================================
# Dockerfile — HunyuanOCR + LoRA | RunPod Serverless
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Alias python3 → python
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# ── Cài uv ───────────────────────────────────────────────────
RUN pip install uv -q

# ── Tạo venv tại /app/.venv ──────────────────────────────────
RUN uv venv /app/.venv

# Đặt PATH để mọi lệnh RUN sau đều dùng venv này
ENV PATH="/app/.venv/bin:$PATH"

# ── Cài vLLM vào venv ────────────────────────────────────────
RUN uv pip install -U vllm --torch-backend auto

# ── Cài transformers đúng commit HunyuanOCR yêu cầu ─────────
RUN pip install -q \
    "git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4" \
    huggingface_hub peft pillow openai

# ── Copy startup script ──────────────────────────────────────
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

CMD ["/app/start.sh"]
