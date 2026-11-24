# service.py
import hmac
import os

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io, torch
from transformers import AutoImageProcessor, SiglipForImageClassification

# === Load model 1 lần khi khởi động ===
MODEL_NAME = "strangerguardhf/nsfw_image_detection"
model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

id2label = {
    "0": "Anime Picture",
    "1": "Hentai",
    "2": "Normal",
    "3": "Pornography",
    "4": "Enticing or Sensual"
}

LABELS = [id2label[str(i)] for i in range(len(id2label))]

app = FastAPI(title="nsfw-image-detection", version="1.0.0")

@app.get("/health")
def health():
    return {"ok": True}

def classify(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    scores = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}
    # Chính sách ví dụ (tuỳ chỉnh theo nhu cầu)
    is_nsfw = (
        scores.get("Pornography", 0) >= 0.50 or
        scores.get("Hentai", 0)      >= 0.50 or
        scores.get("Enticing or Sensual", 0) >= 0.70
    )
    # gợi ý nhãn cao nhất
    top_label = max(scores, key=scores.get)
    return {"scores": scores, "top_label": top_label, "is_nsfw": is_nsfw}

async def require_api_key(request: Request):
    key = os.getenv("GATEWAY_API_KEY")
    if not key:
        raise HTTPException(503, "Set GATEWAY_API_KEY before using the NSFW service")
    authorization = request.headers.get("authorization", "")
    if not hmac.compare_digest(authorization.encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "Invalid service credentials")


@app.post("/analyze", dependencies=[Depends(require_api_key)])
async def analyze(image: UploadFile = File(...)):
    data = await image.read()
    try:
        result = classify(data)
        return JSONResponse({"ok": True, **result})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
