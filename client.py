"""
OCR sổ đỏ — client gọi vLLM OpenAI-compatible server
Ref: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
"""

import base64
import json
import re
from pathlib import Path

from openai import OpenAI  # pip install openai

# ===== CHỈNH SỬA =====
RUNPOD_POD_ID = "YOUR_POD_ID"        # từ RunPod dashboard
MODEL_NAME    = "so-do-ocr"           # khớp với LORA_NAME trong start.sh
MAX_TOKENS    = 16384                  # output dài cho sổ đỏ
# ======================

BASE_URL = f"https://{RUNPOD_POD_ID}-8000.proxy.runpod.net/v1"

client = OpenAI(
    base_url=BASE_URL,
    api_key="EMPTY",   # vLLM không cần api-key thật (trừ khi set --api-key)
)


# ── Helpers ──────────────────────────────────────────────────

def encode_image(image_path: str) -> tuple[str, str]:
    """Đọc ảnh → base64 + mime type."""
    suffix = Path(image_path).suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".webp": "image/webp"}.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return b64, mime


def clean_output(text: str) -> str:
    """HunyuanOCR đôi khi lặp chuỗi cuối — hàm này dọn sạch."""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count, i = 0, n - length
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]
    return text


# ── Prompt templates cho sổ đỏ ───────────────────────────────

PROMPTS = {
    # Extract toàn bộ nội dung, giữ cấu trúc markdown
    "full": (
        "提取文档图片中正文的所有信息用markdown格式表示，"
        "其中页眉、页脚部分忽略，表格用html格式表达，"
        "按照阅读顺序组织进行解析。"
    ),
    # Text + tọa độ bounding box từng dòng
    "spotting": "检测并识别图片中的文字，将文本坐标格式化输出。",

    # Extract JSON các trường quan trọng của sổ đỏ
    "json": (
        "提取图片中的以下字段内容，并按照JSON格式返回："
        "['số thửa', 'số tờ bản đồ', 'diện tích', 'địa chỉ thửa đất',"
        " 'mục đích sử dụng', 'thời hạn sử dụng', 'tên người sử dụng',"
        " 'số vào sổ cấp GCN', 'ngày tháng năm cấp', 'nơi cấp']"
        "\nChỉ trả về JSON, không giải thích thêm."
    ),

    # Chỉ lấy text thuần, nhanh nhất
    "text_only": "提取图中的文字。",
}


# ── Hàm OCR chính ────────────────────────────────────────────

def ocr_so_do(image_path: str, task: str = "json") -> dict:
    """
    OCR ảnh sổ đỏ qua vLLM OpenAI-compatible API.

    Args:
        image_path: đường dẫn file ảnh (.jpg / .png / .webp)
        task: "json" | "full" | "spotting" | "text_only"

    Returns:
        dict với keys: raw_text, structured (nếu task=json), usage
    """
    b64, mime = encode_image(image_path)
    prompt_text = PROMPTS.get(task, PROMPTS["full"])

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
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
        temperature=0,
        max_tokens=MAX_TOKENS,
    )

    raw = response.choices[0].message.content
    text = clean_output(raw)

    result = {
        "task": task,
        "image": image_path,
        "raw_text": text,
        "usage": {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        },
    }

    # Parse JSON nếu task là "json"
    if task == "json":
        try:
            match = re.search(r"\{[\s\S]*\}", text)
            result["structured"] = json.loads(match.group()) if match else None
        except (json.JSONDecodeError, AttributeError):
            result["structured"] = None

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


# ── Chạy thử ─────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if not health_check():
        print("\nServer chưa sẵn sàng. Thử lại sau 3-5 phút.")
        sys.exit(1)

    list_models()
    print()

    image = "so_do.jpg"   # ← thay bằng ảnh thật của bạn

    # --- Task 1: Extract JSON các trường sổ đỏ ---
    print("=" * 50)
    print("TASK: JSON extract")
    r = ocr_so_do(image, task="json")
    if r["structured"]:
        print(json.dumps(r["structured"], ensure_ascii=False, indent=2))
    else:
        print("Raw:", r["raw_text"])
    print("Tokens:", r["usage"])

    # --- Task 2: Full markdown ---
    print("\n" + "=" * 50)
    print("TASK: Full markdown")
    r2 = ocr_so_do(image, task="full")
    print(r2["raw_text"])
