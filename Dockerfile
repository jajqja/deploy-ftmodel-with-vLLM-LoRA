# Dockerfile — HunyuanOCR + LoRA deploy trên RunPod
# Base: PyTorch với CUDA 12.1 (tương thích vLLM nightly)

FROM runpod/pytorch:2.3.0-py3.11-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# uv (package manager nhanh hơn pip)
RUN pip install uv -q

# Transformers phiên bản HunyuanOCR yêu cầu
RUN pip install -q \
    "git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4" \
    huggingface_hub peft pillow requests

# vLLM nightly (bắt buộc để support HunyuanOCR)
RUN uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly --system -q

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

CMD ["/app/start.sh"]
