"""
Client gọi vLLM OpenAI-compatible server
Ref: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
"""

import base64
from pathlib import Path

from openai import OpenAI  # pip install openai

RUNPOD_POD_ID = "YOUR_POD_ID"        # từ RunPod dashboard
MODEL_NAME    = "newai-ocr"           # khớp với LORA_NAME trong start.sh
MAX_TOKENS    = 5000                  # output dài cho sổ đỏ

BASE_URL = f"https://{RUNPOD_POD_ID}-8000.proxy.runpod.net/v1"
API_KEY = "sk_YOUR_KEY_HERE"

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

def encode_image(image_path: str) -> tuple[str, str]:
    """Đọc ảnh → base64 + mime type."""
    suffix = Path(image_path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".webp": "image/webp"}.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return b64, mime


# ── Prompt templates cho sổ đỏ ───────────────────────────────

PROMPTS = "Extract all information from the main body of a Vietnam Certificate of Land Use Rights image (Giấy chứng nhận quyền sử dụng đất của Việt Nam). Present any tables in Markdown format. Replace non-text elements such as land plot diagrams, seals, signatures, or images with [short descriptions in Vietnamese enclosed in brackets]. Ensure the parsing is accurate, preserves the original meaning, and is organized clearly following the natural reading order of the document."


def predict(image_path: str) -> dict:
    b64, mime = encode_image(image_path)

    # Chat completions — đúng format OpenAI multimodal
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": PROMPTS},
                ],
            },
        ],
        temperature=0,
        top_p=1,
        max_tokens=MAX_TOKENS,
        extra_body={
            "top_k": 1,
            "repetition_penalty": 1.03
        },
    )

    raw = response.choices[0].message.content

    result = {
        "image": image_path,
        "raw_text": raw,
        "usage": {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        },
    }

    return result


# ── Utility ──────────────────────────────────────────────────

def health_check() -> bool:
    """Kiểm tra server sẵn sàng chưa."""
    import requests
    try:
        url = BASE_URL.replace("/v1", "") + "/health"
        r = requests.get(url, timeout=10)
        ok = r.status_code == 200
        print("✓ Server healthy" if ok else f"✗ Status: {r.status_code}")
        return ok
    except Exception as e:
        print(f"✗ Server chưa sẵn sàng: {e}")
        return False


def list_models():
    """Liệt kê model + LoRA adapter đang chạy."""
    models = client.models.list()
    print("Models available:")
    for m in models.data:
        print(f"  • {m.id}")


# ── Chạy thử ──

if __name__ == "__main__":
    import sys

    if not health_check():
        print("\nServer chưa sẵn sàng. Thử lại sau 3-5 phút.")
        sys.exit(1)

    list_models()
    print()

    image = "002.png"

    print("=" * 50)
    res = predict(image)
    print("Raw:", res["raw_text"])
    print("Tokens:", res["usage"])
    print("=" * 50)
